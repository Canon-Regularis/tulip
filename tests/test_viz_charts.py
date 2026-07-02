"""Tests for tulip.viz.charts (matplotlib / plotly, extra ``viz``)."""

from __future__ import annotations

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import ClassProbability, Prediction
from tulip.viz.charts import confusion_matrix_heatmap, probability_bar_chart


def _prediction() -> Prediction:
    return Prediction(
        label="podhale",
        probabilities=(
            ClassProbability(label="podhale", probability=0.55),
            ClassProbability(label="silesia", probability=0.30),
            ClassProbability(label="kurpie", probability=0.15),
        ),
    )


def test_bar_chart_matplotlib_returns_figure_with_sorted_bars() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    fig = probability_bar_chart(_prediction(), backend="matplotlib")
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    assert len(ax.patches) == 3
    widths = [patch.get_width() for patch in ax.patches]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] == pytest.approx(55.0)
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert labels[0] == "Podhale"
    assert "Podhale" in ax.get_title()


def test_bar_chart_matplotlib_top_k() -> None:
    pytest.importorskip("matplotlib")
    fig = probability_bar_chart(_prediction(), top_k=2)
    assert len(fig.axes[0].patches) == 2


def test_bar_chart_matplotlib_threshold_line() -> None:
    pytest.importorskip("matplotlib")
    fig = probability_bar_chart(_prediction(), abstain_threshold=0.5)
    ax = fig.axes[0]
    vlines = [line for line in ax.lines if np.allclose(line.get_xdata(), [50.0, 50.0])]
    assert len(vlines) == 1


def test_bar_chart_matplotlib_abstained_title() -> None:
    pytest.importorskip("matplotlib")
    prediction = Prediction(
        label=None,
        abstained=True,
        probabilities=(ClassProbability(label="podhale", probability=0.2),),
    )
    fig = probability_bar_chart(prediction)
    assert "abstained" in fig.axes[0].get_title()


def test_bar_chart_plotly_backend() -> None:
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    fig = probability_bar_chart(_prediction(), backend="plotly", abstain_threshold=0.4)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].orientation == "h"
    assert list(fig.data[0].x) == pytest.approx([55.0, 30.0, 15.0])


def test_bar_chart_validation_errors() -> None:
    with pytest.raises(ConfigurationError):
        probability_bar_chart(_prediction(), top_k=0)
    with pytest.raises(ConfigurationError):
        probability_bar_chart(_prediction(), backend="gnuplot")
    empty = Prediction(label=None, probabilities=())
    with pytest.raises(DataError):
        probability_bar_chart(empty)


def test_confusion_heatmap_matplotlib() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matrix = [[8, 1, 1], [2, 7, 1], [0, 2, 8]]
    fig = confusion_matrix_heatmap(matrix, ["podhale", "silesia", "kurpie"])
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert labels == ["podhale", "silesia", "kurpie"]


def test_confusion_heatmap_normalized_rows() -> None:
    pytest.importorskip("matplotlib")
    matrix = [[2, 2], [0, 0]]  # second row sums to zero: must not divide by zero
    fig = confusion_matrix_heatmap(matrix, ["a", "b"], normalize=True)
    image = fig.axes[0].images[0]
    values = np.asarray(image.get_array())
    assert values[0].tolist() == pytest.approx([0.5, 0.5])
    assert values[1].tolist() == pytest.approx([0.0, 0.0])


def test_confusion_heatmap_plotly() -> None:
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    fig = confusion_matrix_heatmap([[3, 1], [0, 4]], ["a", "b"], backend="plotly")
    assert isinstance(fig, go.Figure)


def test_confusion_heatmap_validation_errors() -> None:
    with pytest.raises(DataError):
        confusion_matrix_heatmap([[1, 2, 3], [4, 5, 6]], ["a", "b"])
    with pytest.raises(DataError):
        confusion_matrix_heatmap([[1, 2], [3, 4]], ["a", "b", "c"])
    with pytest.raises(ConfigurationError):
        confusion_matrix_heatmap([[1]], ["a"], backend="seaborn")
