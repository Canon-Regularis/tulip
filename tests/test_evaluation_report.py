"""Tests for tulip.evaluation.report.EvaluationReport (markdown, JSON persistence)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.report import ClassMetrics, EvaluationReport

if TYPE_CHECKING:
    from pathlib import Path


class TestMetadataRoundTrip:
    def test_json_native_metadata_round_trips_exactly(self, tmp_path: Path) -> None:
        original = compute_metrics(
            ["a", "b"], ["a", "b"], metadata={"model": "nb", "seeds": [1, 2]}
        )
        original.save(tmp_path / "r.json")
        assert EvaluationReport.load(tmp_path / "r.json") == original

    def test_a_tuple_in_metadata_comes_back_as_a_list(self, tmp_path: Path) -> None:
        """`metadata` is free-form and serialised as JSON, so containers normalise.

        `load()` used to promise equality with the saved instance unconditionally.
        It cannot: JSON has no tuple. The docstring now says so; this pins the
        actual behaviour so nobody re-asserts the false contract.
        """
        original = compute_metrics(["a", "b"], ["a", "b"], metadata={"pair": (1, 2)})
        original.save(tmp_path / "r.json")
        loaded = EvaluationReport.load(tmp_path / "r.json")

        assert loaded != original
        assert original.metadata["pair"] == (1, 2)
        assert loaded.metadata["pair"] == [1, 2]


@pytest.fixture
def report() -> EvaluationReport:
    return compute_metrics(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
        y_proba=[[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]],
        metadata={"model": "naive_bayes", "split": "test"},
    )


def test_json_round_trip_equality(report: EvaluationReport, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"
    report.save(path)
    assert path.is_file()
    loaded = EvaluationReport.load(path)
    assert loaded == report
    assert loaded.roc_auc_macro_ovr == pytest.approx(0.75)
    assert loaded.confusion == ((1, 1), (0, 2))


def test_round_trip_preserves_none_roc_auc(tmp_path: Path) -> None:
    report = compute_metrics(["a", "b"], ["a", "b"])
    path = tmp_path / "report.json"
    report.save(path)
    assert EvaluationReport.load(path).roc_auc_macro_ovr is None


def test_to_markdown_contains_summary_and_per_class_rows(report: EvaluationReport) -> None:
    markdown = report.to_markdown()
    assert "naive_bayes" in markdown
    assert "| Accuracy | 0.7500 |" in markdown
    assert "| F1 (macro) |" in markdown
    assert "## Per-class metrics" in markdown
    # One row per class, aligned to the report's precision/recall/f1/support.
    assert "| a | 1.0000 | 0.5000 | 0.6667 | 2 |" in markdown
    assert "| b | 0.6667 | 1.0000 | 0.8000 | 2 |" in markdown


def test_to_markdown_renders_missing_auc_as_na() -> None:
    report = compute_metrics(["a", "b"], ["a", "b"])
    assert "| ROC AUC (macro OVR) | n/a |" in report.to_markdown()


def test_summary_line_is_compact_and_prefixed(report: EvaluationReport) -> None:
    line = report.summary_line()
    assert "\n" not in line
    assert line.startswith("naive_bayes/test: ")
    assert "acc=0.7500" in line
    assert "f1_macro=" in line
    assert "n=4" in line


def test_summary_line_without_metadata_has_no_prefix() -> None:
    line = compute_metrics(["a", "b"], ["a", "b"]).summary_line()
    assert line.startswith("n=2 ")
    assert "auc=n/a" in line


def test_report_is_frozen(report: EvaluationReport) -> None:
    with pytest.raises(ValidationError):
        report.accuracy = 0.0  # type: ignore[misc]


def test_misaligned_confusion_rejected() -> None:
    per_class = {"a": ClassMetrics(precision=1.0, recall=1.0, f1=1.0, support=1)}
    with pytest.raises(ValidationError, match="confusion"):
        EvaluationReport(
            accuracy=1.0,
            balanced_accuracy=1.0,
            precision_macro=1.0,
            recall_macro=1.0,
            f1_macro=1.0,
            precision_weighted=1.0,
            recall_weighted=1.0,
            f1_weighted=1.0,
            labels=("a",),
            per_class=per_class,
            confusion=((1, 0), (0, 1)),  # 2x2 for a single label
            n_samples=1,
        )


def test_per_class_keys_must_match_labels() -> None:
    per_class = {"b": ClassMetrics(precision=1.0, recall=1.0, f1=1.0, support=1)}
    with pytest.raises(ValidationError, match="per_class"):
        EvaluationReport(
            accuracy=1.0,
            balanced_accuracy=1.0,
            precision_macro=1.0,
            recall_macro=1.0,
            f1_macro=1.0,
            precision_weighted=1.0,
            recall_weighted=1.0,
            f1_weighted=1.0,
            labels=("a",),
            per_class=per_class,
            confusion=((1,),),
            n_samples=1,
        )
