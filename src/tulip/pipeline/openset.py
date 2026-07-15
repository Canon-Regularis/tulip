"""Open-set novelty detection over a conformal classifier.

A closed-set classifier always names one of its known dialects, even for an
input from a gwara it never trained on or from a different language entirely.
Open-set detection adds an honest "none of the above". It reuses the conformal
threshold: a class is in the prediction set when its nonconformity is at or below
the calibrated cutoff, so a row whose every class is excluded conforms to no
known dialect and is flagged novel.

The novelty test is a deterministic quantile, like the conformal set it rests on.
The ground truth for evaluation is built in: a test sample whose gold dialect is
not in the training vocabulary is genuinely novel, which is exactly the
deployment question of meeting a new region.

This composes over a fitted
:class:`~tulip.pipeline.conformal.ConformalClassifier`, like the calibrated and
conformal wrappers compose over the base, and never subclasses it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import DataError
from tulip.evaluation._format import format_metric, markdown_table
from tulip.pipeline._assembly import raws_for_task
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from tulip.core.types import Sample
    from tulip.labels.taxonomy import LabelLevel
    from tulip.pipeline.classifier import DialectClassifier
    from tulip.pipeline.conformal import ConformalClassifier

__all__ = ["OpenSetClassifier", "OpenSetPrediction", "OpenSetReport"]

_logger = get_logger(__name__)


class OpenSetPrediction(BaseModel):
    """One open-set decision: a known label or a novelty flag."""

    model_config = ConfigDict(frozen=True)

    in_distribution: bool
    novelty_score: float = Field(ge=0.0, le=1.0)
    prediction_set: tuple[str, ...]
    top_label: str


class OpenSetReport(BaseModel):
    """Open-set quality on a test set mixing known and unseen-dialect samples."""

    model_config = ConfigDict(frozen=True)

    model: str
    target: str
    alpha: float = Field(gt=0.0, lt=1.0)
    n_known: int = Field(ge=0)
    n_novel: int = Field(ge=0)
    known_coverage: float = Field(ge=0.0, le=1.0)
    false_novelty_rate: float = Field(ge=0.0, le=1.0)
    detection_rate: float | None = None
    novelty_auroc: float | None = None

    def to_markdown(self) -> str:
        """Render the open-set metrics as a small markdown table."""
        rows = [
            ("known coverage", format_metric(self.known_coverage)),
            ("false-novelty rate (known flagged novel)", format_metric(self.false_novelty_rate)),
            ("detection rate (novel flagged novel)", format_metric(self.detection_rate)),
            ("novelty AUROC", format_metric(self.novelty_auroc)),
        ]
        title = f"# Open-set - {self.model} ({self.target})"
        note = f"alpha={self.alpha}, {self.n_known} known and {self.n_novel} novel samples"
        return f"{title}\n\n{note}\n\n{markdown_table(('Metric', 'Value'), rows)}"


class OpenSetClassifier:
    """Flag inputs unlike any known dialect, on top of a conformal classifier.

    Args:
        conformal: A fitted
            :class:`~tulip.pipeline.conformal.ConformalClassifier`.
    """

    def __init__(self, conformal: ConformalClassifier) -> None:
        self.conformal = conformal

    @property
    def base(self) -> DialectClassifier:
        """The underlying fitted classifier."""
        return self.conformal.base

    @property
    def classes_(self) -> tuple[str, ...]:
        """Known class vocabulary, delegated to the base classifier."""
        return self.base.classes_

    @property
    def target(self) -> LabelLevel:
        """Target label granularity, delegated to the base classifier."""
        return self.base.target

    def predict_openset(self, raws: Sequence[Any]) -> list[OpenSetPrediction]:
        """Return one open-set decision per raw input.

        A row is in-distribution when at least one class conforms (a non-empty
        conformal set). ``novelty_score`` is the top nonconformity ``1 - max p``,
        so a higher score means a less familiar input.
        """
        thresholds = self.conformal.thresholds()
        proba = self.base.predict_proba(raws)
        classes = self.base.classes_
        predictions: list[OpenSetPrediction] = []
        for row in proba:
            nonconformity = 1.0 - row
            included = tuple(
                classes[index]
                for index in np.argsort(row)[::-1]  # most probable first
                if nonconformity[index] <= thresholds[index]
            )
            top_index = int(np.argmax(row))
            predictions.append(
                OpenSetPrediction(
                    in_distribution=bool(included),
                    novelty_score=float(1.0 - row[top_index]),
                    prediction_set=included,
                    top_label=classes[top_index],
                )
            )
        return predictions

    def predict_openset_for(self, samples: Sequence[Sample]) -> list[OpenSetPrediction]:
        """Open-set decisions for samples, read via the base's modality.

        Raises:
            DataError: if any sample lacks the base classifier's input modality.
        """
        return self.predict_openset(raws_for_task(samples, self.base.task))

    def evaluate(self, samples: Sequence[Sample]) -> OpenSetReport:
        """Measure open-set quality on labelled TEST samples.

        A sample is treated as novel when its gold dialect is not in the training
        vocabulary. Reports coverage on the known part (it should hold near
        ``1 - alpha``), the false-novelty rate on the known part, the detection
        rate on the novel part, and the novelty AUROC.

        Raises:
            DataError: if no sample carries the modality and a label.
        """
        batch = self.base.labelled_batch(samples)
        if not batch.raws:
            raise DataError(
                f"no usable samples for target {self.base.target.value!r} "
                f"(skipped {batch.n_skipped}); open-set evaluation needs labelled data"
            )
        predictions = self.predict_openset(batch.raws)
        known = set(self.base.classes_)
        is_novel = [label not in known for label in batch.labels]

        known_total = known_covered = false_novel = 0
        novel_total = detected = 0
        for prediction, label, novel in zip(predictions, batch.labels, is_novel, strict=True):
            if novel:
                novel_total += 1
                detected += not prediction.in_distribution
            else:
                known_total += 1
                known_covered += label in prediction.prediction_set
                false_novel += not prediction.in_distribution

        return OpenSetReport(
            model=self.base.model_config.name,
            target=self.base.target.value,
            alpha=self.conformal.alpha,
            n_known=known_total,
            n_novel=novel_total,
            known_coverage=known_covered / known_total if known_total else 0.0,
            false_novelty_rate=false_novel / known_total if known_total else 0.0,
            detection_rate=detected / novel_total if novel_total else None,
            novelty_auroc=_novelty_auroc([p.novelty_score for p in predictions], is_novel),
        )


def _novelty_auroc(scores: Sequence[float], is_novel: Sequence[bool]) -> float | None:
    """AUROC of the novelty score against the true-novelty labels, or ``None``.

    Returns ``None`` unless both a known and a novel sample are present, since
    AUROC is undefined for a single class.
    """
    if len(set(is_novel)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(is_novel, dtype=int), np.asarray(scores, dtype=float)))
