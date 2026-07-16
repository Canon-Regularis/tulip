"""Reliability diagram: per-bin accuracy versus confidence.

Split out of :mod:`tulip.viz.charts` because it is the one chart that reads a
calibration report, and therefore the only one that imports
:mod:`tulip.evaluation`. Keeping it here means the probability-bar and
confusion-matrix charts, which need only a :class:`~tulip.core.types.Prediction`
or a plain matrix, no longer pull the evaluation layer in transitively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.core.exceptions import DataError
from tulip.evaluation.calibration import reliability_curve
from tulip.utils.optional import optional_import
from tulip.viz._common import (
    _BAR_COLOR,
    _GRIDLINE,
    _INK_MUTED,
    _INK_SECONDARY,
    _WINNER_COLOR,
    _validate_backend,
)

if TYPE_CHECKING:
    import numpy as np

    from tulip.evaluation.calibration import CalibrationReport

__all__ = ["reliability_diagram"]


def reliability_diagram(
    report: CalibrationReport,
    *,
    backend: str = "matplotlib",
    title: str = "Reliability diagram",
) -> Any:
    """Chart calibration: per-bin accuracy versus confidence against the ideal.

    Two stacked panels share a confidence x-axis. The upper panel plots each
    bin's observed accuracy against its mean confidence together with the
    ``y = x`` perfect-calibration diagonal; points below the line are
    overconfident, above it underconfident. The lower panel is a histogram of
    how many samples fell in each bin, so a reliability point backed by three
    samples is visibly distinct from one backed by three hundred. The measured
    ECE and MCE are annotated on the diagram.

    Args:
        report: A report from
            :func:`~tulip.evaluation.calibration.compute_calibration`.
        backend: ``"matplotlib"`` or ``"plotly"``.
        title: Chart title.

    Returns:
        A ``matplotlib.figure.Figure`` or ``plotly.graph_objects.Figure``.

    Raises:
        ConfigurationError: If the backend is unknown.
        DataError: If the report has no populated bins to chart.
        MissingDependencyError: If the backend library is missing (extra ``viz``).
    """
    key = _validate_backend(backend)
    confidence, accuracy, count = reliability_curve(report)
    if confidence.size == 0:
        raise DataError("calibration report has no populated bins to chart")
    if key == "matplotlib":
        return _matplotlib_reliability(report, confidence, accuracy, count, title)
    return _plotly_reliability(report, confidence, accuracy, count, title)


def _matplotlib_reliability(
    report: CalibrationReport,
    confidence: np.ndarray,
    accuracy: np.ndarray,
    count: np.ndarray,
    title: str,
) -> Any:
    figure_mod = optional_import("matplotlib.figure", extra="viz", purpose="reliability diagrams")
    fig = figure_mod.Figure(figsize=(6.4, 6.4))
    ax_rel, ax_hist = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Perfect calibration is the diagonal; the model's curve sits off it.
    ax_rel.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color=_INK_MUTED,
        linewidth=1.2,
        label="perfect calibration",
        zorder=2,
    )
    ax_rel.plot(
        confidence,
        accuracy,
        marker="o",
        color=_WINNER_COLOR,
        linewidth=1.6,
        label="model",
        zorder=3,
    )
    ax_rel.set_xlim(0.0, 1.0)
    ax_rel.set_ylim(0.0, 1.0)
    ax_rel.set_ylabel("Accuracy", color=_INK_SECONDARY)
    ax_rel.set_title(title)
    ax_rel.legend(loc="upper left", frameon=False)
    ax_rel.grid(color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax_rel.annotate(
        f"ECE {report.ece:.3f}   MCE {report.mce:.3f}   Brier {report.brier:.3f}",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        color=_INK_SECONDARY,
    )

    width = 0.9 / max(report.n_bins, 1)
    ax_hist.bar(confidence, count, width=width, color=_BAR_COLOR, zorder=3)
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("Confidence", color=_INK_SECONDARY)
    ax_hist.set_ylabel("Count", color=_INK_SECONDARY)
    ax_hist.grid(axis="y", color=_GRIDLINE, linewidth=0.8, zorder=0)

    for ax in (ax_rel, ax_hist):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=_INK_SECONDARY)
    fig.tight_layout()
    return fig


def _plotly_reliability(
    report: CalibrationReport,
    confidence: np.ndarray,
    accuracy: np.ndarray,
    count: np.ndarray,
    title: str,
) -> Any:
    go = optional_import("plotly.graph_objects", extra="viz", purpose="reliability diagrams")
    subplots = optional_import("plotly.subplots", extra="viz", purpose="reliability diagrams")
    fig = subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line={"dash": "dash", "color": _INK_MUTED},
            name="perfect calibration",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(confidence),
            y=list(accuracy),
            mode="lines+markers",
            line={"color": _WINNER_COLOR},
            marker={"color": _WINNER_COLOR},
            name="model",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=list(confidence),
            y=[int(value) for value in count],
            marker={"color": _BAR_COLOR},
            name="count",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(range=[0.0, 1.0], row=1, col=1)
    fig.update_xaxes(range=[0.0, 1.0], title_text="Confidence", row=2, col=1)
    fig.update_yaxes(range=[0.0, 1.0], title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_layout(
        title=f"{title}: ECE {report.ece:.3f}, MCE {report.mce:.3f}",
        template="plotly_white",
    )
    return fig
