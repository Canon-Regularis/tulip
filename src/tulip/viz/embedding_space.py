"""2-D projection of sample or dialect embeddings for cluster visualisation.

:func:`project_embeddings` reduces a feature/embedding matrix to two
dimensions with t-SNE (scikit-learn, always available) or UMAP (optional
extra ``umap``), returning a tidy ``(x, y, label)`` DataFrame.
:func:`plot_embedding_space` scatters that frame coloured by label: the
dialect-similarity / clustering visual.

Determinism: both projectors are driven by the explicit ``seed`` argument
(t-SNE additionally uses PCA initialisation), so the same inputs and seed
reproduce the same layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tulip.core.exceptions import DataError
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import
from tulip.viz._common import _BACKENDS, _validate_choice

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

#: Column names of the DataFrame produced by :func:`project_embeddings`.
PROJECTION_COLUMNS: tuple[str, str, str] = ("x", "y", "label")

_METHODS = ("tsne", "umap")

#: Fixed-order categorical palette (validated set). Hues are assigned to the
#: sorted labels in this order and never cycled on their own: labels beyond
#: the eighth reuse hues but switch marker shape, so identity is carried by
#: the colour+shape pair rather than colour alone.
_CATEGORICAL_PALETTE: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
)
_MPL_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
_PLOTLY_MARKERS: tuple[str, ...] = (
    "circle",
    "square",
    "triangle-up",
    "diamond",
    "triangle-down",
    "cross",
    "x",
    "star",
)


def _as_dense_2d(data: Any) -> np.ndarray:
    """Coerce array-like or scipy-sparse input to a dense 2-D float matrix."""
    if hasattr(data, "toarray"):  # scipy sparse matrices from TF-IDF pipelines
        data = data.toarray()
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2:
        raise DataError(f"expected a 2-D embedding matrix, got shape {matrix.shape}")
    return matrix


def _tsne_projection(matrix: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE  # core dependency; imported here to keep import cheap

    n_samples, n_features = matrix.shape
    # sklearn requires perplexity < n_samples; the usual default (30) breaks on
    # small corpora, so scale it down while keeping it >= 1 where possible.
    perplexity = min(max(1.0, min(30.0, (n_samples - 1) / 3.0)), float(n_samples - 1))
    init = "pca" if n_features >= 2 else "random"  # PCA init needs >= 2 features
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init=init,
        learning_rate="auto",
        random_state=seed,
    )
    return np.asarray(tsne.fit_transform(matrix), dtype=np.float64)


def _umap_projection(matrix: np.ndarray, seed: int) -> np.ndarray:
    umap_mod = optional_import("umap", extra="umap", purpose="UMAP embedding projection")
    # UMAP requires n_neighbors < n_samples; no floor of 2 here, or the
    # smallest accepted input (n=2) would violate that and trigger UMAP's
    # silent self-truncation warning.
    n_neighbors = min(15, matrix.shape[0] - 1)
    reducer = umap_mod.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=seed)
    return np.asarray(reducer.fit_transform(matrix), dtype=np.float64)


def project_embeddings(
    X: Any,
    labels: Sequence[str],
    *,
    method: str = "tsne",
    seed: int = 42,
) -> pd.DataFrame:
    """Project an embedding matrix to two dimensions for visualisation.

    Args:
        X: Array-like or scipy-sparse matrix of shape ``(n_samples, n_features)``.
        labels: One label per row of ``X`` (dialect, family, cluster id, ...).
        method: ``"tsne"`` (scikit-learn, always available) or ``"umap"``
            (optional extra ``umap``).
        seed: Random seed; identical inputs and seed give identical layouts.

    Returns:
        A DataFrame with columns :data:`PROJECTION_COLUMNS`: float ``x`` and
        ``y`` coordinates plus the string ``label`` per sample.

    Raises:
        ConfigurationError: If ``method`` is unknown.
        DataError: If ``X`` is not 2-D, does not match ``labels`` in length,
            or has fewer than two rows.
        MissingDependencyError: If ``method="umap"`` and umap-learn is missing.
    """
    method_key = _validate_choice(method, _METHODS, "projection method")
    matrix = _as_dense_2d(X)
    n_samples = matrix.shape[0]
    if len(labels) != n_samples:
        raise DataError(f"got {len(labels)} labels for {n_samples} embedding rows")
    if n_samples < 2:
        raise DataError("embedding projection needs at least two samples")

    logger.debug("projecting %d x %d embeddings with %s", *matrix.shape, method_key)
    if method_key == "tsne":
        coordinates = _tsne_projection(matrix, seed)
    else:
        coordinates = _umap_projection(matrix, seed)
    return pd.DataFrame(
        {
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "label": [str(label) for label in labels],
        }
    )


def _label_styles(labels: Sequence[str], markers: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """Assign each sorted unique label a fixed ``(colour, marker)`` pair.

    Hues follow the fixed categorical order; once the eight hues are used, the
    next block of labels reuses them with the next marker shape so no two
    classes share both colour and shape.
    """
    unique = sorted(set(labels))
    styles: dict[str, tuple[str, str]] = {}
    for index, label in enumerate(unique):
        color = _CATEGORICAL_PALETTE[index % len(_CATEGORICAL_PALETTE)]
        marker = markers[(index // len(_CATEGORICAL_PALETTE)) % len(markers)]
        styles[label] = (color, marker)
    return styles


def plot_embedding_space(
    df: pd.DataFrame,
    *,
    backend: str = "matplotlib",
    title: str = "Dialect embedding space",
) -> Any:
    """Scatter a projected embedding frame, coloured by label, with a legend.

    Args:
        df: A DataFrame with the :data:`PROJECTION_COLUMNS` columns, as
            produced by :func:`project_embeddings`.
        backend: ``"matplotlib"`` or ``"plotly"``.
        title: Chart title.

    Returns:
        A ``matplotlib.figure.Figure`` or ``plotly.graph_objects.Figure``.

    Raises:
        ConfigurationError: If the backend is unknown.
        DataError: If required columns are missing.
        MissingDependencyError: If the backend library is missing (extra ``viz``).
    """
    backend_key = _validate_choice(backend, _BACKENDS, "plot backend")
    missing = [column for column in PROJECTION_COLUMNS if column not in df.columns]
    if missing:
        raise DataError(f"embedding frame is missing columns: {', '.join(missing)}")

    if backend_key == "matplotlib":
        return _matplotlib_scatter(df, title)
    return _plotly_scatter(df, title)


def _matplotlib_scatter(df: pd.DataFrame, title: str) -> Any:
    figure_mod = optional_import("matplotlib.figure", extra="viz", purpose="embedding-space plots")
    fig = figure_mod.Figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111)
    styles = _label_styles(df["label"].tolist(), _MPL_MARKERS)
    for label, (color, marker) in styles.items():
        subset = df[df["label"] == label]
        ax.scatter(
            subset["x"],
            subset["y"],
            s=36,
            c=color,
            marker=marker,
            label=label,
            alpha=0.85,
            edgecolors="white",  # surface ring keeps overlapping points separable
            linewidths=0.5,
        )
    # Projection axes carry no interpretable units; keep the frame quiet.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title="label",
    )
    fig.tight_layout()
    return fig


def _plotly_scatter(df: pd.DataFrame, title: str) -> Any:
    go = optional_import("plotly.graph_objects", extra="viz", purpose="embedding-space plots")
    fig = go.Figure()
    styles = _label_styles(df["label"].tolist(), _PLOTLY_MARKERS)
    for label, (color, marker) in styles.items():
        subset = df[df["label"] == label]
        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                name=label,
                marker={
                    "color": color,
                    "symbol": marker,
                    "size": 9,
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=f"{label}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"showticklabels": False, "title": ""},
        yaxis={"showticklabels": False, "title": ""},
        legend={"title": "label"},
    )
    return fig
