"""Tests for tulip.viz.embedding_space (t-SNE path is pure scikit-learn)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.viz.embedding_space import PROJECTION_COLUMNS, plot_embedding_space, project_embeddings


def _random_embeddings(n: int = 24, dim: int = 8, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim))
    labels = [f"dialect-{i % 3}" for i in range(n)]
    return embeddings, labels


def test_tsne_projection_shape_and_columns() -> None:
    embeddings, labels = _random_embeddings()
    df = project_embeddings(embeddings, labels, method="tsne", seed=42)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(PROJECTION_COLUMNS)
    assert len(df) == len(labels)
    assert df["x"].dtype.kind == "f"
    assert df["y"].dtype.kind == "f"
    assert set(df["label"]) == {"dialect-0", "dialect-1", "dialect-2"}


def test_tsne_projection_deterministic_per_seed() -> None:
    embeddings, labels = _random_embeddings()
    first = project_embeddings(embeddings, labels, seed=42)
    second = project_embeddings(embeddings, labels, seed=42)
    np.testing.assert_allclose(first[["x", "y"]].to_numpy(), second[["x", "y"]].to_numpy())


def test_tsne_projection_tiny_sample_perplexity_guard() -> None:
    embeddings, labels = _random_embeddings(n=4)
    df = project_embeddings(embeddings, labels, method="tsne", seed=7)
    assert len(df) == 4  # default perplexity (30) would raise without the guard


def test_projection_accepts_sparse_input() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")
    embeddings, labels = _random_embeddings(n=12, dim=6)
    df = project_embeddings(scipy_sparse.csr_matrix(embeddings), labels, seed=3)
    assert len(df) == 12


def test_projection_label_mismatch_raises() -> None:
    embeddings, labels = _random_embeddings()
    with pytest.raises(DataError):
        project_embeddings(embeddings, labels[:-1])


def test_projection_rejects_1d_input() -> None:
    with pytest.raises(DataError):
        project_embeddings(np.zeros(10), ["a"] * 10)


def test_projection_rejects_single_sample() -> None:
    with pytest.raises(DataError):
        project_embeddings(np.zeros((1, 4)), ["a"])


def test_projection_unknown_method_raises() -> None:
    embeddings, labels = _random_embeddings(n=6)
    with pytest.raises(ConfigurationError):
        project_embeddings(embeddings, labels, method="pca")


def test_umap_projection_shape() -> None:
    pytest.importorskip("umap")
    embeddings, labels = _random_embeddings(n=12, dim=6)
    df = project_embeddings(embeddings, labels, method="umap", seed=42)
    assert list(df.columns) == list(PROJECTION_COLUMNS)
    assert len(df) == 12


def test_plot_embedding_space_matplotlib() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    embeddings, labels = _random_embeddings(n=15, dim=4)
    df = project_embeddings(embeddings, labels, seed=42)
    fig = plot_embedding_space(df, backend="matplotlib")
    assert isinstance(fig, matplotlib.figure.Figure)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    legend_labels = {text.get_text() for text in legend.get_texts()}
    assert legend_labels == {"dialect-0", "dialect-1", "dialect-2"}


def test_plot_embedding_space_plotly() -> None:
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    embeddings, labels = _random_embeddings(n=15, dim=4)
    df = project_embeddings(embeddings, labels, seed=42)
    fig = plot_embedding_space(df, backend="plotly")
    assert isinstance(fig, go.Figure)
    assert {trace.name for trace in fig.data} == {"dialect-0", "dialect-1", "dialect-2"}


def test_plot_embedding_space_missing_columns_raises() -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(DataError):
        plot_embedding_space(pd.DataFrame({"x": [0.0], "y": [1.0]}))


def test_plot_embedding_space_unknown_backend_raises() -> None:
    df = pd.DataFrame({"x": [0.0], "y": [1.0], "label": ["a"]})
    with pytest.raises(ConfigurationError):
        plot_embedding_space(df, backend="bokeh")
