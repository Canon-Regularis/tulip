"""Risk-coverage (selective-prediction) evaluation of confidence thresholds.

The pipeline already *ships* abstention:
:class:`~tulip.pipeline.classifier.DialectClassifier` abstains when its top
probability falls below ``abstain_threshold``. But nothing measures the
trade-off that choice makes, so a threshold is picked blind. This module scores
it: sweep the confidence threshold from "answer everything" down to "answer only
the surest few" and, at each coverage, report the selective *risk* (error rate
among the answered). From that curve come three operator-facing numbers:

* **AURC**: area under the risk-coverage curve; a single lower-is-better summary
  of how well the model's confidence ranks its own correctness.
* **accuracy @ coverage**: "if I answer the most-confident 90%, how accurate am
  I?", the number you set an SLA against.
* **coverage @ error**: "to stay under 10% error, how much can I answer?".

A model whose confidence is meaningless has a flat curve (AURC ≈ base error); a
model that knows when it is right earns accuracy far above its base rate at low
coverage. The curve is a deterministic function of the per-sample confidences
and correctness (ties broken by sample order), so a committed selective report
regenerates byte-for-byte.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation._format import format_metric, markdown_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.evaluation.predictions import SplitPredictions

__all__ = [
    "DEFAULT_COVERAGE_TARGETS",
    "SelectivePoint",
    "SelectiveReport",
    "risk_coverage_curve",
    "selective_report",
]

#: Coverage levels reported by default (fractions of samples answered).
DEFAULT_COVERAGE_TARGETS = (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)


class SelectivePoint(BaseModel):
    """One point on the risk-coverage curve.

    Attributes:
        coverage: Fraction of samples answered (the most-confident ones).
        risk: Error rate among the answered samples (``1 - accuracy``).
        threshold: The lowest confidence that is still answered at this coverage.
    """

    model_config = ConfigDict(frozen=True)

    coverage: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)

    @property
    def accuracy(self) -> float:
        """Accuracy among the answered samples (``1 - risk``)."""
        return 1.0 - self.risk


class SelectiveReport(BaseModel):
    """Selective-prediction summary for one model on one split.

    Attributes:
        model: Model name, for labelling.
        split: Split name.
        n_samples: Number of scored samples.
        aurc: Area under the risk-coverage curve (lower is better).
        base_error: Selective risk at full coverage (the plain error rate).
        targets: Risk/accuracy at each requested coverage level.
        curve: The full risk-coverage curve, one point per sample-prefix.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    split: str
    n_samples: int = Field(ge=1)
    aurc: float = Field(ge=0.0, le=1.0)
    base_error: float = Field(ge=0.0, le=1.0)
    targets: tuple[SelectivePoint, ...]
    curve: tuple[SelectivePoint, ...]

    def accuracy_at_coverage(self, coverage: float) -> float:
        """Accuracy among the most-confident ``coverage`` fraction of samples."""
        return _point_at_coverage(self.curve, coverage).accuracy

    def coverage_at_error(self, max_error: float) -> float:
        """Largest coverage whose selective risk stays at or below ``max_error``.

        Returns ``0.0`` when even answering only the single most-confident
        sample already exceeds ``max_error``.
        """
        answerable = [point.coverage for point in self.curve if point.risk <= max_error + 1e-12]
        return max(answerable) if answerable else 0.0

    def to_markdown(self) -> str:
        """Render the report as markdown: a summary plus a per-target table."""
        summary_rows = [
            ("Samples", str(self.n_samples)),
            ("AURC", format_metric(self.aurc)),
            ("Base accuracy", format_metric(1.0 - self.base_error)),
        ]
        target_rows = [
            (
                format_metric(point.coverage, digits=2),
                format_metric(point.accuracy),
                format_metric(point.risk),
                format_metric(point.threshold),
            )
            for point in self.targets
        ]
        title = f"# Selective prediction: {self.model} ({self.split})"
        return "\n\n".join(
            [
                title,
                markdown_table(("Metric", "Value"), summary_rows),
                "## Accuracy at coverage",
                markdown_table(("Coverage", "Accuracy", "Risk", "Min confidence"), target_rows),
            ]
        )


def risk_coverage_curve(
    correct: Sequence[bool], confidence: Sequence[float]
) -> list[SelectivePoint]:
    """Build the risk-coverage curve from per-sample correctness and confidence.

    Samples are answered most-confident first; ties in confidence are broken by
    original order, so the curve is deterministic. Point ``k`` (1-indexed)
    answers the ``k`` most-confident samples: its coverage is ``k / n`` and its
    risk is the error rate within that prefix.

    Args:
        correct: Whether each sample's prediction was right.
        confidence: Each sample's top predicted probability, aligned with
            ``correct``.

    Returns:
        One :class:`SelectivePoint` per sample, in ascending coverage order.

    Raises:
        ConfigurationError: if the inputs differ in length or are empty.
    """
    correct_array = np.asarray(correct, dtype=bool)
    confidence_array = np.asarray(confidence, dtype=float)
    if correct_array.shape != confidence_array.shape:
        raise ConfigurationError(
            f"correct and confidence must be the same length; got {correct_array.shape} "
            f"and {confidence_array.shape}"
        )
    n = int(correct_array.shape[0])
    if n == 0:
        raise ConfigurationError("cannot build a risk-coverage curve on zero samples")

    # Most-confident first; ties broken by original index (ascending) for
    # determinism. lexsort's last key is primary, so negate confidence.
    order = np.lexsort((np.arange(n), -confidence_array))
    ranked_correct = correct_array[order]
    ranked_confidence = confidence_array[order]

    k = np.arange(1, n + 1)
    cumulative_correct = np.cumsum(ranked_correct)
    selective_risk = 1.0 - cumulative_correct / k
    coverage = k / n
    return [
        SelectivePoint(
            coverage=float(coverage[i]),
            risk=float(selective_risk[i]),
            threshold=float(ranked_confidence[i]),
        )
        for i in range(n)
    ]


def selective_report(
    predictions: SplitPredictions,
    *,
    coverage_targets: Sequence[float] = DEFAULT_COVERAGE_TARGETS,
) -> SelectiveReport:
    """Score a model's abstention trade-off from its per-sample predictions.

    Args:
        predictions: Per-sample records for one model on one split.
        coverage_targets: Coverage levels to report accuracy/risk at.

    Returns:
        A frozen :class:`SelectiveReport`.

    Raises:
        ConfigurationError: if there are no records, or a target is outside
            ``(0, 1]``.
    """
    if len(predictions) == 0:
        raise ConfigurationError("cannot build a selective report on zero predictions")
    for target in coverage_targets:
        if not 0.0 < target <= 1.0:
            raise ConfigurationError(f"coverage target must be in (0, 1], got {target}")

    curve = risk_coverage_curve(predictions.correct().tolist(), predictions.confidences().tolist())
    # Uniform coverage steps of 1/n, so the mean selective risk is the discrete
    # area under the risk-coverage curve.
    aurc = float(np.mean([point.risk for point in curve]))
    targets = tuple(_point_at_coverage(curve, target) for target in coverage_targets)
    return SelectiveReport(
        model=predictions.model,
        split=predictions.split,
        n_samples=len(predictions),
        aurc=aurc,
        base_error=curve[-1].risk,
        targets=targets,
        curve=tuple(curve),
    )


def _point_at_coverage(curve: Sequence[SelectivePoint], coverage: float) -> SelectivePoint:
    """Return the curve point answering the most-confident ``coverage`` fraction.

    Coverage maps to answering ``round(coverage * n)`` samples (at least one), so
    the same fraction always selects the same prefix regardless of ``n``.
    """
    n = len(curve)
    index = min(n, max(1, round(coverage * n))) - 1
    return curve[index]
