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

from tulip.core.exceptions import ConfigurationError
from tulip.pipeline._assembly import (
    _BaseDelegating,
    conformal_row_sets,
    raws_for_task,
    require_labelled_batch,
    scored_in_vocab_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Self

    from tulip.core.types import Sample
    from tulip.pipeline.protocols import CalibratableClassifier

__all__ = ["ConformalClassifier", "ConformalPrediction", "ConformalReport"]


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


class ConformalClassifier(_BaseDelegating):
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
        base: CalibratableClassifier,
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
        batch = require_labelled_batch(
            self.base, samples, context="conformal needs held-out labelled data"
        )
        proba, y_index = scored_in_vocab_rows(
            self.base,
            batch,
            log_label="conformal",
            empty_error=(
                "calibration set has no samples whose gold label is known to the base "
                "classifier; cannot compute conformal scores"
            ),
        )
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
        for included, top_index, row in conformal_row_sets(proba, classes, thresholds):
            top_label = classes[top_index]
            # A conformal set is never empty; fall back to the point estimate.
            predictions.append(
                ConformalPrediction(
                    prediction_set=included if included else (top_label,),
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
        batch = require_labelled_batch(
            self.base, samples, context="conformal needs held-out labelled data"
        )
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

    def thresholds(self) -> np.ndarray:
        """Per-class nonconformity thresholds aligned to ``classes_`` (fit first).

        A class ``c`` is in the conformal set when ``1 - p(c) <= thresholds[c]``.
        Exposed for open-set detection, where a row whose every class is excluded
        is unlike any known dialect.

        Raises:
            ConfigurationError: if called before :meth:`fit_conformal`.
        """
        return self._thresholds()

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

    def _raws_of(self, samples: Sequence[Sample]) -> list[Any]:
        """Extract the base's raw inputs, erroring on a missing modality."""
        return raws_for_task(samples, self.base.task)

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
