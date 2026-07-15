"""Fairness: subgroup disparity across slices, with unpaired significance.

A pooled score hides the failure mode this benchmark exists to expose: a model
that nails one corpus and fails a rare gwara, or that works for the speakers it
saw and not the ones it did not. This turns the per-slice metrics into a
disparity summary. For each slice dimension it reports the best and worst group,
the accuracy gap between them, and the min-over-max ratio.

The gap is between two disjoint groups, so the test of whether it is real is an
unpaired two-proportion z-test, not the paired McNemar the model-vs-model
significance uses. The normal tail comes from :func:`math.erfc`, so no SciPy is
added. The per-dimension p-values are Holm-corrected together, reusing the same
correction as the significance report. A worst group below the support floor is
flagged and never headlines: its gap is real but noisy, and its test rarely
clears significance.

Everything is pure over a :class:`~tulip.evaluation.predictions.SplitPredictions`,
so a report is deterministic and byte-stable, like the error report it extends.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tulip.evaluation.error_analysis import DEFAULT_LOW_SUPPORT, slice_metrics

if TYPE_CHECKING:
    from tulip.evaluation.error_analysis import SliceMetric
    from tulip.evaluation.predictions import SplitPredictions

__all__ = ["DisparityMetric", "FairnessReport", "fairness_report", "worst_group_gap"]


class DisparityMetric(BaseModel):
    """Best-versus-worst group accuracy for one slice dimension."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    n_groups: int = Field(ge=2)
    best_group: str
    best_accuracy: float = Field(ge=0.0, le=1.0)
    worst_group: str
    worst_accuracy: float = Field(ge=0.0, le=1.0)
    gap: float = Field(ge=0.0, le=1.0)
    ratio: float = Field(ge=0.0, le=1.0)
    worst_low_support: bool
    p_value: float | None = None
    significant: bool = False


class FairnessReport(BaseModel):
    """Subgroup disparity across every slice dimension."""

    model_config = ConfigDict(frozen=True)

    model: str
    split: str
    alpha: float = Field(gt=0.0, lt=1.0)
    dimensions: tuple[DisparityMetric, ...]

    @property
    def max_gap(self) -> float:
        """Largest gap among dimensions whose worst group has adequate support.

        Falls back to the overall largest gap when every worst group is
        low-support, and to ``0.0`` when there are no dimensions.
        """
        reliable = [d.gap for d in self.dimensions if not d.worst_low_support]
        pool = reliable or [d.gap for d in self.dimensions]
        return max(pool) if pool else 0.0

    def to_markdown(self) -> str:
        """Render the disparities as a markdown table, widest gap first."""
        from tulip.evaluation._format import format_metric, markdown_table

        rows = [
            (
                metric.dimension,
                metric.worst_group + (" (low support)" if metric.worst_low_support else ""),
                format_metric(metric.worst_accuracy),
                metric.best_group,
                format_metric(metric.best_accuracy),
                format_metric(metric.gap),
                format_metric(metric.ratio),
                "yes" if metric.significant else "no",
            )
            for metric in sorted(self.dimensions, key=lambda m: (-m.gap, m.dimension))
        ]
        title = f"# Fairness - {self.model} ({self.split})"
        note = f"Largest reliable subgroup gap: {format_metric(self.max_gap)}"
        headers = (
            "Dimension",
            "Worst group",
            "Worst acc",
            "Best group",
            "Best acc",
            "Gap",
            "Ratio",
            "Sig",
        )
        return f"{title}\n\n{note}\n\n{markdown_table(headers, rows)}"


def fairness_report(
    predictions: SplitPredictions,
    *,
    low_support: int = DEFAULT_LOW_SUPPORT,
    alpha: float = 0.05,
) -> FairnessReport:
    """Compute best-versus-worst subgroup disparity for every slice dimension.

    Args:
        predictions: The per-sample predictions to disaggregate.
        low_support: Group size below which a group is flagged low-support.
        alpha: Significance level for the Holm-corrected gap tests.

    Returns:
        A :class:`FairnessReport`, one :class:`DisparityMetric` per dimension that
        has at least two groups.
    """
    by_dimension: dict[str, list[SliceMetric]] = {}
    for group in slice_metrics(predictions, low_support=low_support):
        by_dimension.setdefault(group.dimension, []).append(group)

    pending: list[tuple[str, SliceMetric, SliceMetric, float]] = []
    for dimension in sorted(by_dimension):
        groups = by_dimension[dimension]
        if len(groups) < 2:
            continue
        worst = min(groups, key=lambda g: (g.accuracy, g.value))
        best = min(groups, key=lambda g: (-g.accuracy, g.value))
        p_value = _two_proportion_p(
            round(best.accuracy * best.n), best.n, round(worst.accuracy * worst.n), worst.n
        )
        pending.append((dimension, best, worst, p_value))

    adjusted = _holm([p for _, _, _, p in pending])
    dimensions = tuple(
        DisparityMetric(
            dimension=dimension,
            n_groups=len(by_dimension[dimension]),
            best_group=best.value,
            best_accuracy=best.accuracy,
            worst_group=worst.value,
            worst_accuracy=worst.accuracy,
            gap=best.accuracy - worst.accuracy,
            ratio=worst.accuracy / best.accuracy if best.accuracy > 0.0 else 0.0,
            worst_low_support=worst.low_support,
            p_value=p_value,
            significant=adjusted_p < alpha,
        )
        for (dimension, best, worst, p_value), adjusted_p in zip(pending, adjusted, strict=True)
    )
    return FairnessReport(
        model=predictions.model, split=predictions.split, alpha=alpha, dimensions=dimensions
    )


def worst_group_gap(report: FairnessReport) -> DisparityMetric | None:
    """The single most severe disparity: the largest gap with a reliable worst group.

    Falls back to the overall widest gap when every worst group is low-support,
    and to ``None`` when the report has no dimensions.
    """
    if not report.dimensions:
        return None
    reliable = [d for d in report.dimensions if not d.worst_low_support]
    pool = reliable or list(report.dimensions)
    return max(pool, key=lambda d: (d.gap, d.dimension))


def _two_proportion_p(correct_a: int, n_a: int, correct_b: int, n_b: int) -> float:
    """Two-sided unpaired two-proportion z-test p-value (normal approximation)."""
    if n_a == 0 or n_b == 0:
        return 1.0
    pooled = (correct_a + correct_b) / (n_a + n_b)
    if pooled in (0.0, 1.0):
        return 1.0  # no variation to separate the groups
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if standard_error == 0.0:
        return 1.0
    z = (correct_a / n_a - correct_b / n_b) / standard_error
    return math.erfc(abs(z) / math.sqrt(2.0))


def _holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjustment, reusing the significance module's rule."""
    from tulip.evaluation.significance import _holm as holm

    return holm(p_values)
