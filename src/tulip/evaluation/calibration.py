"""Confidence-calibration metrics for a classifier's probability estimates.

A model can top the leaderboard on macro-F1 yet still be badly *overconfident*:
when it says 0.9 it may be right only 0.6 of the time. That gap matters because
:class:`~tulip.pipeline.classifier.DialectClassifier` abstains on a threshold
over these very probabilities, a cutoff on a number that does not mean what it
looks like unless the model is calibrated.

This module reports that gap without touching the standard metrics:

* :class:`CalibrationBin` / :class:`CalibrationReport` are frozen value objects
  (they only *hold* results; :func:`compute_calibration` produces them).
* :func:`compute_calibration` computes top-label Expected/Maximum Calibration
  Error and the multiclass Brier score, binning confidences by a pluggable
  ``strategy`` selected through a dict factory rather than an ``if``/``elif``
  chain, so a new binning scheme is added without editing the consumer.

It deliberately imports nothing from :mod:`tulip.evaluation.report`: the report
holds an optional :class:`CalibrationReport`, so the dependency runs one way and
there is no import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation._format import format_metric, markdown_table

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "CalibrationBin",
    "CalibrationReport",
    "compute_calibration",
    "reliability_curve",
]

#: Slack allowed when range-checking probabilities, so a value that is 1.0 only
#: up to floating-point round-off is not rejected.
_RANGE_TOL = 1e-9


class CalibrationBin(BaseModel):
    """One confidence bin: how sure the model was versus how often it was right.

    Attributes:
        lower: Inclusive-ish lower edge of the bin's confidence interval.
        upper: Upper edge of the bin's confidence interval.
        confidence: Mean top-label probability of the samples in the bin.
        accuracy: Fraction of those samples whose top label was correct.
        count: Number of samples that fell in the bin (always ``> 0``; empty
            bins are dropped from a report).
    """

    model_config = ConfigDict(frozen=True)

    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=1)


class CalibrationReport(BaseModel):
    """Calibration of one classifier's probabilities on one split.

    ``ece`` and ``mce`` are the standard *top-label* calibration errors: samples
    are grouped by their maximum predicted probability and, per bin, the mean
    confidence is compared with the observed accuracy. ``brier`` is the
    multiclass Brier score, which ranges over ``[0, 2]`` (not ``[0, 1]``): the
    extra unit comes from summing the squared error across every class column.

    Attributes:
        n_bins: Number of bins *requested*; ``len(bins)`` may be smaller because
            empty bins are dropped.
        n_samples: Number of scored samples.
        ece: Expected Calibration Error in ``[0, 1]``; the sample-weighted mean
            absolute gap between bin confidence and bin accuracy.
        mce: Maximum Calibration Error in ``[0, 1]``; the worst such gap.
        brier: Multiclass Brier score in ``[0, 2]`` (lower is better).
        bins: The non-empty bins, in ascending confidence order.
        strategy: The binning strategy used (``"uniform"`` or ``"quantile"``).
    """

    model_config = ConfigDict(frozen=True)

    n_bins: int = Field(ge=1)
    n_samples: int = Field(ge=1)
    ece: float = Field(ge=0.0, le=1.0)
    mce: float = Field(ge=0.0, le=1.0)
    brier: float = Field(ge=0.0, le=2.0)
    bins: tuple[CalibrationBin, ...]
    strategy: str

    def to_markdown(self) -> str:
        """Render the report as markdown: a summary table plus per-bin reliability.

        Returns:
            A markdown document (heading, summary metrics table, per-bin table)
            ending without a trailing newline.
        """
        summary_rows = [
            ("Samples", str(self.n_samples)),
            ("Bins (requested)", str(self.n_bins)),
            ("Bins (populated)", str(len(self.bins))),
            ("Strategy", self.strategy),
            ("ECE", format_metric(self.ece)),
            ("MCE", format_metric(self.mce)),
            ("Brier", format_metric(self.brier)),
        ]
        bin_rows = [
            (
                f"[{bin_.lower:.3f}, {bin_.upper:.3f}]",
                format_metric(bin_.confidence),
                format_metric(bin_.accuracy),
                str(bin_.count),
            )
            for bin_ in self.bins
        ]
        return "\n\n".join(
            [
                "# Calibration report",
                markdown_table(("Metric", "Value"), summary_rows),
                "## Per-bin reliability",
                markdown_table(("Bin", "Confidence", "Accuracy", "Count"), bin_rows),
            ]
        )


def _uniform_edges(confidences: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin edges spanning the whole ``[0, 1]`` probability range."""
    return np.linspace(0.0, 1.0, n_bins + 1)


def _quantile_edges(confidences: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bin edges: the confidence quantiles, anchored to ``[0, 1]``.

    Duplicate edges (from ties in the confidence distribution) are collapsed, so
    a degenerate distribution simply yields fewer bins rather than empty ones.
    """
    edges = np.quantile(confidences, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    return np.unique(edges)


#: Confidence-binning strategies, keyed by the name a caller (or a YAML) passes.
#: A dict factory keeps :func:`compute_calibration` closed to modification: a new
#: strategy is a new entry here, not a new branch in the consumer.
_BIN_STRATEGIES: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "uniform": _uniform_edges,
    "quantile": _quantile_edges,
}


def compute_calibration(
    y_true: Sequence[Any],
    y_proba: Any,
    labels: Sequence[Any],
    *,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> CalibrationReport:
    """Measure how well predicted probabilities match observed accuracy.

    Args:
        y_true: Gold labels, one per sample. Values are coerced to ``str``.
        y_proba: Probability matrix of shape ``(n_samples, n_labels)`` whose
            columns are ordered like ``labels`` and each row is a class
            distribution over those labels.
        labels: Class order fixing the meaning of ``y_proba``'s columns.
        n_bins: Number of confidence bins (equal-width, or equal-count for the
            ``"quantile"`` strategy).
        strategy: ``"uniform"`` (equal-width bins) or ``"quantile"`` (equal-count
            bins). Selected through a dict factory.

    Returns:
        A frozen :class:`CalibrationReport` with top-label ECE/MCE, the
        multiclass Brier score, and the non-empty confidence bins.

    Raises:
        ConfigurationError: If ``n_bins < 1``; the strategy is unknown; the input
            is empty; ``labels`` has duplicates; ``y_proba`` is not a numeric 2-D
            array; its row or column count disagrees with the samples or labels;
            or a label in ``y_true`` is absent from ``labels``.
    """
    if n_bins < 1:
        raise ConfigurationError(f"n_bins must be >= 1, got {n_bins}")
    if strategy not in _BIN_STRATEGIES:
        expected = ", ".join(sorted(_BIN_STRATEGIES))
        raise ConfigurationError(
            f"unknown calibration strategy {strategy!r}; expected one of {expected}"
        )

    true_list = [str(value) for value in y_true]
    if not true_list:
        raise ConfigurationError("cannot compute calibration on zero samples")

    label_list = [str(value) for value in labels]
    if len(set(label_list)) != len(label_list):
        raise ConfigurationError(f"labels contains duplicates: {label_list!r}")

    proba = _as_proba_matrix(y_proba, n_samples=len(true_list), n_labels=len(label_list))

    label_to_index = {label: index for index, label in enumerate(label_list)}
    missing = sorted(set(true_list) - set(label_to_index))
    if missing:
        raise ConfigurationError(f"labels observed in y_true are missing from `labels`: {missing}")
    true_indices = np.array([label_to_index[value] for value in true_list])

    predicted = np.argmax(proba, axis=1)
    confidences = proba[np.arange(len(true_list)), predicted]
    correct = (predicted == true_indices).astype(float)
    brier = _brier(proba, true_indices)

    edges = _BIN_STRATEGIES[strategy](confidences, n_bins)
    # ``right=True`` puts a value that lands exactly on an interior edge into the
    # lower bin; values outside [0, 1] clamp into the first/last bin.
    bin_of = np.digitize(confidences, edges[1:-1], right=True)

    bins, ece, mce = _summarise_bins(edges, bin_of, confidences, correct)
    return CalibrationReport(
        n_bins=n_bins,
        n_samples=len(true_list),
        ece=ece,
        mce=mce,
        brier=brier,
        bins=bins,
        strategy=strategy,
    )


def reliability_curve(report: CalibrationReport) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-bin arrays for plotting a reliability diagram.

    Args:
        report: A report from :func:`compute_calibration`.

    Returns:
        A ``(confidence, accuracy, count)`` triple of parallel arrays, one entry
        per non-empty bin in ascending confidence order. A perfectly calibrated
        model has ``accuracy == confidence`` in every bin (the diagonal).
    """
    confidence = np.array([bin_.confidence for bin_ in report.bins], dtype=float)
    accuracy = np.array([bin_.accuracy for bin_ in report.bins], dtype=float)
    count = np.array([bin_.count for bin_ in report.bins], dtype=int)
    return confidence, accuracy, count


def _as_proba_matrix(y_proba: Any, *, n_samples: int, n_labels: int) -> np.ndarray:
    """Coerce ``y_proba`` to a validated ``(n_samples, n_labels)`` float matrix."""
    try:
        proba = np.asarray(y_proba, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("y_proba must be a numeric 2-D probability matrix") from exc
    if proba.ndim != 2:
        raise ConfigurationError(f"y_proba must be 2-D, got ndim={proba.ndim}")
    if proba.shape[0] != n_samples:
        raise ConfigurationError(
            f"y_proba has {proba.shape[0]} rows but there are {n_samples} samples"
        )
    if proba.shape[1] != n_labels:
        raise ConfigurationError(
            f"y_proba has {proba.shape[1]} columns but there are {n_labels} labels"
        )
    # Validate the VALUES too, not just the shape. A NaN row makes argmax select
    # the NaN slot, and the bin's mean confidence then fails CalibrationBin's
    # Field(ge=0, le=1), surfacing as a cryptic pydantic error about an
    # internal value object rather than about the caller's bad probabilities.
    if not np.all(np.isfinite(proba)):
        raise ConfigurationError("y_proba contains non-finite values (NaN or inf)")
    if proba.size:
        low, high = float(proba.min()), float(proba.max())
        if low < -_RANGE_TOL or high > 1.0 + _RANGE_TOL:
            raise ConfigurationError(
                f"y_proba values must lie in [0, 1]; got range [{low:.6g}, {high:.6g}]"
            )
    return proba


def _brier(proba: np.ndarray, true_indices: np.ndarray) -> float:
    """Multiclass Brier score: ``mean_i sum_k (p_ik - 1[y_i == k])^2`` in ``[0, 2]``."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(proba.shape[0]), true_indices] = 1.0
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    # Clamp only against floating-point drift past the theoretical [0, 2] bound.
    return min(2.0, max(0.0, brier))


def _summarise_bins(
    edges: np.ndarray,
    bin_of: np.ndarray,
    confidences: np.ndarray,
    correct: np.ndarray,
) -> tuple[tuple[CalibrationBin, ...], float, float]:
    """Reduce per-sample confidences/correctness to non-empty bins plus ECE/MCE.

    Empty bins contribute nothing to ECE (their sample weight is zero) and are
    omitted from the returned tuple, so dropping them cannot corrupt the totals.
    """
    n = len(confidences)
    bins: list[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    for index in range(len(edges) - 1):
        mask = bin_of == index
        count = int(mask.sum())
        if count == 0:
            continue
        # Clamp against floating-point drift past [0, 1]: _as_proba_matrix accepts a
        # top probability up to 1 + _RANGE_TOL, which would otherwise push a bin's
        # mean confidence past the CalibrationBin bound and raise.
        confidence = min(1.0, max(0.0, float(confidences[mask].mean())))
        accuracy = min(1.0, max(0.0, float(correct[mask].mean())))
        gap = abs(accuracy - confidence)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append(
            CalibrationBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                confidence=confidence,
                accuracy=accuracy,
                count=count,
            )
        )
    # Clamp only against floating-point drift past the theoretical [0, 1] bound.
    return tuple(bins), min(1.0, max(0.0, ece)), min(1.0, max(0.0, mce))
