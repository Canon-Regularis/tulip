"""Split (inductive) conformal prediction: calibrated prediction sets.

A classifier's top-1 label is a point estimate with no honest error bar.
Conformal prediction adds one: it returns a *set* of labels (for example
``{podhale, spisz}``) with a distribution-free guarantee that the true label is
in the set at least ``1 - alpha`` of the time. The guarantee needs no assumption
about the model or the data beyond exchangeability of the calibration and test
samples.

This is the split-conformal recipe. A fitted classifier scores a held-out
calibration split. Each calibration sample gets a nonconformity score (here the
Least Ambiguous set-valued Classifier score, ``1 - p(true class)``). The
``1 - alpha`` quantile of those scores, with the finite-sample correction, is the
threshold ``qhat``. At test time the set is every class whose nonconformity is at
or below ``qhat``, i.e. ``{c : p(c) >= 1 - qhat}``.

Two variants:

* **Marginal** (default). One global ``qhat``. Coverage holds on average across
  all classes.
* **Mondrian** (``mondrian=True``). A separate ``qhat`` per class, so coverage
  holds *within* each class. This matters under class imbalance and under the
  speaker-disjoint shift, where a rare dialect can be silently under-covered by a
  single global threshold.

It composes over a fitted :class:`~tulip.pipeline.classifier.DialectClassifier`,
exactly like :class:`~tulip.pipeline.calibrated.CalibratedClassifier`, and never
subclasses it. The frozen core :class:`~tulip.core.types.Prediction` has no
label-set field, so this module returns its own :class:`ConformalPrediction`.
Split conformal is a deterministic quantile, so its output is reproducible.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import TaskType
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Self

    from tulip.core.types import Sample
    from tulip.labels.taxonomy import LabelLevel
    from tulip.pipeline.classifier import DialectClassifier, LabelledBatch

__all__ = ["ConformalClassifier", "ConformalPrediction", "ConformalReport"]

_logger = get_logger(__name__)


class ConformalPrediction(BaseModel):
    """A conformal prediction: a label set plus the point estimate it contains."""

    model_config = ConfigDict(frozen=True)

    prediction_set: tuple[str, ...]
    top_label: str
    top_probability: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(gt=0.0, lt=1.0)

    @property
    def set_size(self) -> int:
        """Number of labels in the set."""
        return len(self.prediction_set)

    def contains(self, label: str) -> bool:
        """Whether ``label`` is in the prediction set."""
        return label in self.prediction_set


class ConformalReport(BaseModel):
    """Empirical coverage and mean set size of a conformal classifier."""

    model_config = ConfigDict(frozen=True)

    n_samples: int = Field(ge=1)
    alpha: float = Field(gt=0.0, lt=1.0)
    mondrian: bool
    coverage: float = Field(ge=0.0, le=1.0)
    mean_set_size: float = Field(ge=0.0)

    @property
    def target_coverage(self) -> float:
        """The coverage the method guarantees (``1 - alpha``)."""
        return 1.0 - self.alpha


class ConformalClassifier:
    """Wrap a fitted classifier to emit calibrated prediction sets.

    Args:
        base: A fitted :class:`~tulip.pipeline.classifier.DialectClassifier`.
        alpha: Miscoverage rate. The set covers the truth at least ``1 - alpha``
            of the time (e.g. ``alpha=0.1`` targets 90% coverage).
        mondrian: Use a per-class threshold (class-conditional coverage) instead
            of one global threshold.

    Raises:
        ConfigurationError: if ``alpha`` is not in ``(0, 1)``.
    """

    def __init__(
        self,
        base: DialectClassifier,
        *,
        alpha: float = 0.1,
        mondrian: bool = False,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ConfigurationError(f"alpha must be within (0, 1), got {alpha}")
        self.base = base
        self.alpha = alpha
        self.mondrian = mondrian

    # ---------------------------------------------------------------- fit

    def fit_conformal(self, samples: Sequence[Sample]) -> Self:
        """Compute the conformal threshold(s) on a HELD-OUT calibration split.

        The samples MUST be data the base classifier never trained on, so the
        calibration and test scores are exchangeable. Rows whose gold label is
        unknown to the base classifier are dropped with a logged count.

        Raises:
            DataError: if the calibration set has no usable, in-vocabulary
                labelled samples.
        """
        proba, y_index = self._scored_calibration(samples)
        # LAC nonconformity: how far the true class is from certainty.
        scores = 1.0 - proba[np.arange(len(y_index)), y_index]
        if self.mondrian:
            global_qhat = _conformal_quantile(scores, self.alpha)
            self.qhat_: dict[str, float] = {}
            for class_index, label in enumerate(self.base.classes_):
                class_scores = scores[y_index == class_index]
                # A class with too few calibration points falls back to the
                # global threshold rather than a wild per-class quantile.
                self.qhat_[label] = (
                    _conformal_quantile(class_scores, self.alpha)
                    if class_scores.size
                    else global_qhat
                )
        else:
            self.global_qhat_: float = _conformal_quantile(scores, self.alpha)
        return self

    # ------------------------------------------------------------ predict

    def predict_set(self, raws: Sequence[Any]) -> list[ConformalPrediction]:
        """Return one prediction set per raw input.

        Raises:
            ConfigurationError: if called before :meth:`fit_conformal`.
        """
        thresholds = self._thresholds()
        proba = self.base.predict_proba(raws)
        classes = self.base.classes_
        predictions: list[ConformalPrediction] = []
        for row in proba:
            nonconformity = 1.0 - row
            included = [
                classes[index]
                for index in np.argsort(row)[::-1]  # most probable first
                if nonconformity[index] <= thresholds[index]
            ]
            top_index = int(np.argmax(row))
            top_label = classes[top_index]
            # A conformal set is never empty; fall back to the point estimate.
            predictions.append(
                ConformalPrediction(
                    prediction_set=tuple(included) if included else (top_label,),
                    top_label=top_label,
                    top_probability=float(row[top_index]),
                    alpha=self.alpha,
                )
            )
        return predictions

    def predict_sets_for(self, samples: Sequence[Sample]) -> list[ConformalPrediction]:
        """Prediction sets for samples, read via the base's modality.

        Raises:
            DataError: if any sample lacks the base classifier's input modality.
        """
        return self.predict_set(self._raws_of(samples))

    def evaluate_coverage(self, samples: Sequence[Sample]) -> ConformalReport:
        """Measure empirical coverage and mean set size on labelled TEST samples.

        Coverage is the fraction of samples whose gold label lands in the
        prediction set. It should sit at or above ``1 - alpha``.

        Raises:
            DataError: if no sample carries the modality and an in-vocabulary
                label.
        """
        batch = self._require_labelled(samples)
        predictions = self.predict_set(batch.raws)
        covered = sum(
            prediction.contains(label)
            for prediction, label in zip(predictions, batch.labels, strict=True)
        )
        sizes = [prediction.set_size for prediction in predictions]
        return ConformalReport(
            n_samples=len(predictions),
            alpha=self.alpha,
            mondrian=self.mondrian,
            coverage=covered / len(predictions),
            mean_set_size=float(np.mean(sizes)),
        )

    # ----------------------------------------------------------- internal

    def _thresholds(self) -> np.ndarray:
        """Per-class thresholds aligned to ``base.classes_`` (global or Mondrian)."""
        if self.mondrian:
            if not hasattr(self, "qhat_"):
                raise ConfigurationError("call fit_conformal before predicting")
            return np.array([self.qhat_[label] for label in self.base.classes_], dtype=float)
        if not hasattr(self, "global_qhat_"):
            raise ConfigurationError("call fit_conformal before predicting")
        return np.full(len(self.base.classes_), self.global_qhat_, dtype=float)

    def _scored_calibration(self, samples: Sequence[Sample]) -> tuple[np.ndarray, np.ndarray]:
        """Probabilities and true-class indices for in-vocabulary calibration rows."""
        batch = self._require_labelled(samples)
        proba = self.base.predict_proba(batch.raws)
        index_of = {label: index for index, label in enumerate(self.base.classes_)}
        kept_rows: list[int] = []
        y_index: list[int] = []
        for row, label in enumerate(batch.labels):
            class_index = index_of.get(label)
            if class_index is not None:
                kept_rows.append(row)
                y_index.append(class_index)
        if not kept_rows:
            raise DataError(
                "calibration set has no samples whose gold label is known to the base "
                "classifier; cannot compute conformal scores"
            )
        dropped = len(batch.labels) - len(kept_rows)
        if dropped:
            _logger.info(
                "conformal: dropped %d/%d rows with labels unseen at training time",
                dropped,
                len(batch.labels),
            )
        return proba[kept_rows], np.asarray(y_index, dtype=int)

    def _require_labelled(self, samples: Sequence[Sample]) -> LabelledBatch:
        """The base's labelled batch, erroring when it is empty."""
        batch = self.base.labelled_batch(samples)
        if not batch.raws:
            raise DataError(
                f"no usable samples for target {self.base.target.value!r} "
                f"(skipped {batch.n_skipped}); conformal needs held-out labelled data"
            )
        return batch

    def _raws_of(self, samples: Sequence[Sample]) -> list[Any]:
        """Extract the base's raw inputs, erroring on a missing modality."""
        task = self.base.task
        raws = [sample.text if task is TaskType.TEXT else sample.audio_path for sample in samples]
        missing = [sample.id for sample, raw in zip(samples, raws, strict=True) if raw is None]
        if missing:
            raise DataError(
                f"{len(missing)} sample(s) carry no {task.value} input and cannot be "
                f"classified (first: {missing[0]!r})"
            )
        return raws

    # ----------------------------------------------------------- delegates

    @property
    def classes_(self) -> tuple[str, ...]:
        """Class-label vocabulary, delegated to the base classifier."""
        return self.base.classes_

    @property
    def target(self) -> LabelLevel:
        """Target label granularity, delegated to the base classifier."""
        return self.base.target

    @property
    def task(self) -> TaskType:
        """Input modality, delegated to the base classifier."""
        return self.base.task

    def __repr__(self) -> str:
        return (
            f"ConformalClassifier(base={self.base!r}, alpha={self.alpha}, mondrian={self.mondrian})"
        )


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The split-conformal threshold: the finite-sample-corrected quantile.

    Returns the ``ceil((n + 1) * (1 - alpha))``-th smallest score. When that rank
    exceeds ``n`` (too few calibration points for the requested coverage), returns
    ``1.0`` so the set includes every class, which is the only way to keep the
    guarantee.
    """
    n = int(scores.shape[0])
    if n == 0:
        return 1.0
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return 1.0
    return float(np.sort(scores)[rank - 1])
