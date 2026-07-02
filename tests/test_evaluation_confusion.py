"""Tests for tulip.evaluation.confusion (arrays, DataFrames, optional plotting)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.confusion import confusion_from_report, plot_confusion, to_dataframe
from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.report import EvaluationReport

Y_TRUE = ["a", "a", "b", "b"]
Y_PRED = ["a", "b", "b", "b"]


@pytest.fixture
def report_ba() -> EvaluationReport:
    """Report with explicit non-sorted label order ["b", "a"]."""
    return compute_metrics(Y_TRUE, Y_PRED, labels=["b", "a"])


def test_confusion_from_report_counts(report_ba: EvaluationReport) -> None:
    matrix = confusion_from_report(report_ba)
    assert matrix.dtype == np.int64
    # Order is ["b", "a"]: row 0 is true "b", column 0 is predicted "b".
    assert matrix.tolist() == [[2, 0], [1, 1]]


def test_to_dataframe_cell_alignment(report_ba: EvaluationReport) -> None:
    frame = to_dataframe(report_ba)
    assert list(frame.index) == ["b", "a"]
    assert list(frame.columns) == ["b", "a"]
    assert frame.index.name == "true"
    assert frame.columns.name == "predicted"
    assert frame.loc["b", "b"] == 2
    assert frame.loc["b", "a"] == 0
    assert frame.loc["a", "b"] == 1
    assert frame.loc["a", "a"] == 1


def test_normalize_true_rows_sum_to_one(report_ba: EvaluationReport) -> None:
    matrix = confusion_from_report(report_ba, normalize="true")
    assert matrix.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert matrix.tolist() == [[1.0, 0.0], [0.5, 0.5]]


def test_normalize_pred_columns_sum_to_one(report_ba: EvaluationReport) -> None:
    matrix = confusion_from_report(report_ba, normalize="pred")
    assert matrix.sum(axis=0) == pytest.approx([1.0, 1.0])
    assert matrix[0, 0] == pytest.approx(2 / 3)  # true b among predicted b


def test_normalize_handles_all_zero_rows_without_nan() -> None:
    # Label "c" is allowed in labels but never occurs: its row/column is all zeros.
    report = compute_metrics(Y_TRUE, Y_PRED, labels=["a", "b", "c"])
    for normalize in ("true", "pred"):
        matrix = confusion_from_report(report, normalize=normalize)
        assert not np.isnan(matrix).any()
        assert matrix[2].tolist() == [0.0, 0.0, 0.0] or matrix[:, 2].tolist() == [0.0, 0.0, 0.0]


def test_invalid_normalize_option_raises(report_ba: EvaluationReport) -> None:
    with pytest.raises(ConfigurationError, match="normalize"):
        confusion_from_report(report_ba, normalize="rows")


def test_plot_confusion_writes_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    report = compute_metrics(Y_TRUE, Y_PRED, metadata={"model": "naive_bayes"})
    out = tmp_path / "plots" / "confusion.png"
    figure = plot_confusion(report, out)
    assert out.is_file()
    assert out.stat().st_size > 0
    assert figure.axes  # the heatmap axes exist on the returned figure


def test_plot_confusion_normalized_without_path(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    report = compute_metrics(Y_TRUE, Y_PRED)
    figure = plot_confusion(report, normalize="true")
    assert figure.axes
    assert not list(tmp_path.iterdir())  # nothing written when path is omitted
