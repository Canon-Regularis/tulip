"""Confusion-matrix views of an :class:`~tulip.evaluation.report.EvaluationReport`.

The report stores raw integer counts; these helpers derive normalised arrays,
labelled pandas DataFrames, and matplotlib figures from them. Rows are true
labels, columns are predicted labels, both in the report's ``labels`` order.
matplotlib is an optional dependency (extra ``viz``) imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tulip.core.exceptions import ConfigurationError
from tulip.utils.io import ensure_dir
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from tulip.evaluation.report import EvaluationReport

NORMALIZE_OPTIONS = ("none", "true", "pred")
_ANNOTATION_LIMIT = 25  # beyond this many classes, cell text becomes unreadable noise


def confusion_from_report(report: EvaluationReport, normalize: str = "none") -> np.ndarray:
    """Return the report's confusion matrix as a numpy array.

    Args:
        report: The evaluation report holding raw confusion counts.
        normalize: ``"none"`` for raw integer counts, ``"true"`` to divide each
            row by its sum (per-true-class recall view), ``"pred"`` to divide
            each column by its sum (per-predicted-class precision view).
            All-zero rows/columns normalise to zeros instead of NaN.

    Returns:
        An ``(n_labels, n_labels)`` array: ``int64`` counts for ``"none"``,
        ``float64`` proportions otherwise.

    Raises:
        ConfigurationError: If ``normalize`` is not one of ``none``/``true``/``pred``.
    """
    if normalize not in NORMALIZE_OPTIONS:
        raise ConfigurationError(
            f"unknown normalize option {normalize!r}; expected one of {NORMALIZE_OPTIONS}"
        )
    counts = np.asarray(report.confusion, dtype=np.int64)
    if normalize == "none":
        return counts
    matrix = counts.astype(np.float64)
    axis = 1 if normalize == "true" else 0
    sums = matrix.sum(axis=axis, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)


def to_dataframe(report: EvaluationReport, normalize: str = "none") -> pd.DataFrame:
    """Return the confusion matrix as a labelled pandas DataFrame.

    Args:
        report: The evaluation report.
        normalize: See :func:`confusion_from_report`.

    Returns:
        A DataFrame indexed by true label (index name ``"true"``) with one
        column per predicted label (columns name ``"predicted"``).
    """
    matrix = confusion_from_report(report, normalize=normalize)
    frame = pd.DataFrame(matrix, index=list(report.labels), columns=list(report.labels))
    frame.index.name = "true"
    frame.columns.name = "predicted"
    return frame


def plot_confusion(
    report: EvaluationReport,
    path: Path | str | None = None,
    *,
    normalize: str = "none",
    cmap: str = "Blues",
    annotate: bool | None = None,
) -> Figure:
    """Plot the confusion matrix as a heatmap and optionally save it.

    Stays readable with many classes: tick labels rotate, the figure grows
    with the label count, and cell annotations switch off automatically past
    25 classes.

    Args:
        report: The evaluation report.
        path: If given, the figure is saved there (PNG/PDF/SVG by extension;
            parent directories are created).
        normalize: See :func:`confusion_from_report`.
        cmap: matplotlib colormap name.
        annotate: Write the value into each cell. Defaults to automatic
            (on for up to 25 classes).

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.

    Raises:
        MissingDependencyError: If matplotlib is not installed
            (``pip install "tulip[viz]"``).
        ConfigurationError: If ``normalize`` is invalid.
    """
    matrix = confusion_from_report(report, normalize=normalize)
    optional_import("matplotlib", extra="viz", purpose="confusion-matrix plotting")
    from matplotlib.figure import Figure

    labels = list(report.labels)
    n = len(labels)
    side = max(5.0, 0.45 * n + 2.5)
    # Figure (not pyplot) keeps us backend-agnostic and free of global state.
    fig = Figure(figsize=(side + 1.0, side))
    ax = fig.add_subplot(1, 1, 1)
    image = ax.imshow(matrix, cmap=cmap, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    title = "Confusion matrix" if normalize == "none" else f"Confusion matrix ({normalize}-norm)"
    if report.metadata.get("model"):
        title += f": {report.metadata['model']}"
    ax.set_title(title)

    if annotate is None:
        annotate = n <= _ANNOTATION_LIMIT
    if annotate:
        threshold = matrix.max() / 2.0 if matrix.size else 0.0
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                text = f"{int(value)}" if normalize == "none" else f"{value:.2f}"
                colour = "white" if value > threshold else "black"
                ax.text(j, i, text, ha="center", va="center", color=colour, fontsize=8)

    fig.tight_layout()
    if path is not None:
        target = Path(path)
        ensure_dir(target.parent)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig


__all__: list[str] = [
    "NORMALIZE_OPTIONS",
    "confusion_from_report",
    "plot_confusion",
    "to_dataframe",
]
