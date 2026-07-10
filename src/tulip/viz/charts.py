"""Probability bar charts and confusion-matrix heatmaps.

Both chart builders support two backends -- ``"matplotlib"`` (returns a
``matplotlib.figure.Figure``) and ``"plotly"`` (returns a
``plotly.graph_objects.Figure``) -- imported lazily via the ``viz`` extra.
Matplotlib figures are constructed directly from ``matplotlib.figure.Figure``
(no pyplot), so no GUI backend or global state is touched and the charts are
safe in headless environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.labels.taxonomy import display_name
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.core.types import Prediction

logger = get_logger(__name__)

_BACKENDS = ("matplotlib", "plotly")

# Palette roles (validated single-hue set): magnitude bars stay in one hue,
# with the winning class emphasised by a darker step, not a new hue.
_BAR_COLOR = "#86b6ef"
_WINNER_COLOR = "#256abf"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"


def _validate_backend(backend: str) -> str:
    key = backend.strip().lower()
    if key not in _BACKENDS:
        raise ConfigurationError(
            f"unknown chart backend {backend!r}; expected one of {', '.join(_BACKENDS)}"
        )
    return key


def probability_bar_chart(
    prediction: Prediction,
    *,
    top_k: int = 10,
    backend: str = "matplotlib",
    abstain_threshold: float | None = None,
    polish_labels: bool = False,
) -> Any:
    """Chart the top-k class probabilities as horizontal bars.

    Bars are sorted descending (winner at the top) on a 0-100 percent axis,
    with the winning class in a darker step of the same hue. When
    ``abstain_threshold`` is given, a dashed reference line marks it so it is
    visible whether the top class cleared the threshold.

    Args:
        prediction: The prediction whose probabilities to chart.
        top_k: How many of the most probable classes to show.
        backend: ``"matplotlib"`` or ``"plotly"``.
        abstain_threshold: Optional abstention threshold in ``[0, 1]`` to mark.
        polish_labels: Use Polish display names instead of English.

    Returns:
        A ``matplotlib.figure.Figure`` or ``plotly.graph_objects.Figure``.

    Raises:
        ConfigurationError: If ``top_k`` < 1 or the backend is unknown.
        DataError: If the prediction carries no probabilities.
        MissingDependencyError: If the backend library is missing (extra ``viz``).
    """
    if top_k < 1:
        raise ConfigurationError(f"top_k must be >= 1, got {top_k}")
    key = _validate_backend(backend)
    entries = prediction.top_k(top_k)
    if not entries:
        raise DataError("prediction has no class probabilities to chart")

    names = [display_name(cp.label, polish=polish_labels) for cp in entries]
    percents = [cp.probability * 100.0 for cp in entries]
    if prediction.abstained:
        colors = [_BAR_COLOR] * len(entries)
        title = "Dialect prediction (abstained)"
    else:
        colors = [_WINNER_COLOR] + [_BAR_COLOR] * (len(entries) - 1)
        top_name = (
            display_name(prediction.label, polish=polish_labels)
            if prediction.label is not None
            else names[0]
        )
        title = f"Dialect prediction: {top_name}"

    if key == "matplotlib":
        return _matplotlib_bars(names, percents, colors, abstain_threshold, title)
    return _plotly_bars(names, percents, colors, abstain_threshold, title)


def _matplotlib_bars(
    names: Sequence[str],
    percents: Sequence[float],
    colors: Sequence[str],
    threshold: float | None,
    title: str,
) -> Any:
    figure_mod = optional_import("matplotlib.figure", extra="viz", purpose="probability bar charts")
    fig = figure_mod.Figure(figsize=(8.0, max(2.4, 0.5 * len(names) + 1.4)))
    ax = fig.add_subplot(111)
    positions = np.arange(len(names))
    ax.barh(positions, list(percents), color=list(colors), height=0.62, zorder=3)
    ax.set_yticks(positions, labels=list(names))
    ax.invert_yaxis()  # winner at the top
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Probability (%)", color=_INK_SECONDARY)
    for position, percent in zip(positions, percents, strict=True):
        ax.annotate(
            f"{percent:.1f}%",
            xy=(percent, position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_INK_SECONDARY,
        )
    if threshold is not None:
        ax.axvline(threshold * 100.0, color=_INK_MUTED, linestyle="--", linewidth=1.2, zorder=2)
        ax.text(
            threshold * 100.0,
            1.02,
            f"abstain threshold ({threshold:.0%})",
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=8,
            color=_INK_MUTED,
        )
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.tick_params(colors=_INK_SECONDARY)
    fig.tight_layout()
    return fig


def _plotly_bars(
    names: Sequence[str],
    percents: Sequence[float],
    colors: Sequence[str],
    threshold: float | None,
    title: str,
) -> Any:
    go = optional_import("plotly.graph_objects", extra="viz", purpose="probability bar charts")
    fig = go.Figure(
        go.Bar(
            x=list(percents),
            y=list(names),
            orientation="h",
            marker={"color": list(colors)},
            text=[f"{p:.1f}%" for p in percents],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"range": [0, 105], "title": "Probability (%)", "ticksuffix": "%"},
        yaxis={"autorange": "reversed"},  # winner at the top
    )
    if threshold is not None:
        fig.add_vline(
            x=threshold * 100.0,
            line_dash="dash",
            line_color=_INK_MUTED,
            annotation_text=f"abstain threshold ({threshold:.0%})",
            annotation_font_color=_INK_MUTED,
        )
    return fig


def confusion_matrix_heatmap(
    matrix: Sequence[Sequence[float]] | Any,
    labels: Sequence[str],
    *,
    backend: str = "matplotlib",
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> Any:
    """Render a confusion matrix as a single-hue heatmap with cell annotations.

    Args:
        matrix: Square array-like of counts, rows = true class, columns =
            predicted class (sklearn convention).
        labels: Class names in matrix order.
        backend: ``"matplotlib"`` or ``"plotly"``.
        normalize: Normalise each row to proportions before plotting (rows
            that sum to zero are left as zeros).
        title: Chart title.

    Returns:
        A ``matplotlib.figure.Figure`` or ``plotly.graph_objects.Figure``.

    Raises:
        ConfigurationError: If the backend is unknown.
        DataError: If the matrix is not square or does not match ``labels``.
        MissingDependencyError: If the backend library is missing (extra ``viz``).
    """
    key = _validate_backend(backend)
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise DataError(f"confusion matrix must be square, got shape {values.shape}")
    if len(labels) != values.shape[0]:
        raise DataError(
            f"got {len(labels)} labels for a {values.shape[0]}x{values.shape[1]} matrix"
        )
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)

    names = [str(label) for label in labels]
    if key == "matplotlib":
        return _matplotlib_confusion(values, names, normalize, title)
    return _plotly_confusion(values, names, normalize, title)


def _matplotlib_confusion(values: np.ndarray, names: list[str], normalize: bool, title: str) -> Any:
    figure_mod = optional_import(
        "matplotlib.figure", extra="viz", purpose="confusion-matrix heatmaps"
    )
    side = max(4.0, 0.6 * len(names) + 2.0)
    fig = figure_mod.Figure(figsize=(side + 1.2, side))
    ax = fig.add_subplot(111)
    image = ax.imshow(values, cmap="Blues", aspect="equal")  # single hue, light -> dark
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(names))
    ax.set_xticks(ticks, labels=names, rotation=45, ha="right")
    ax.set_yticks(ticks, labels=names)
    ax.set_xlabel("Predicted", color=_INK_SECONDARY)
    ax.set_ylabel("True", color=_INK_SECONDARY)
    ax.set_title(title)
    # Annotate every cell; flip to white ink on dark cells for legibility.
    peak = float(values.max()) if values.size else 0.0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text = f"{value:.2f}" if normalize else f"{value:g}"
            ink = "#ffffff" if peak > 0 and value > 0.6 * peak else "#0b0b0b"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=ink)
    fig.tight_layout()
    return fig


def _plotly_confusion(values: np.ndarray, names: list[str], normalize: bool, title: str) -> Any:
    go = optional_import("plotly.graph_objects", extra="viz", purpose="confusion-matrix heatmaps")
    text_format = ".2f" if normalize else "g"
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=names,
            y=names,
            colorscale="Blues",  # single hue, light -> dark
            texttemplate="%{z:" + text_format + "}",
            hovertemplate="true %{y} / predicted %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"title": "Predicted"},
        yaxis={"title": "True", "autorange": "reversed"},  # diagonal top-left -> bottom-right
    )
    return fig
