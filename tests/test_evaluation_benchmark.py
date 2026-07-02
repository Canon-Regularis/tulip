"""Tests for tulip.evaluation.benchmark (comparison tables, JSON persistence)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkResult,
    comparison_table,
    load_benchmark,
    save_benchmark,
    to_markdown_table,
)
from tulip.evaluation.metrics import compute_metrics

EXPECTED_COLUMNS = [
    "model",
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "n_train",
    "wall_seconds",
]


def _result(
    model: str,
    errors: int,
    *,
    seconds: float,
    with_proba: bool = True,
    splits: tuple[str, ...] = ("validation", "test"),
) -> BenchmarkResult:
    """Build a result whose test-split quality decreases with ``errors`` (0..4)."""
    y_true = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y_pred = list(y_true)
    for index in range(errors):  # flip labels from the front
        y_pred[index] = "b" if y_true[index] == "a" else "a"
    proba = None
    if with_proba:
        proba = [[0.9, 0.1] if p == "a" else [0.1, 0.9] for p in y_pred]
    reports = {
        split: compute_metrics(y_true, y_pred, y_proba=proba, metadata={"split": split})
        for split in splits
    }
    return BenchmarkResult(
        experiment="dialect-benchmark",
        model=model,
        target_level="dialect",
        reports=reports,
        wall_seconds=seconds,
        n_train=100,
        n_test=len(y_true),
    )


@pytest.fixture
def results() -> list[BenchmarkResult]:
    return [
        _result("mediocre", 2, seconds=5.0),
        _result("best", 0, seconds=60.0),
        _result("worst", 3, seconds=1.0, with_proba=False),
    ]


def test_comparison_table_columns_and_ordering(results: list[BenchmarkResult]) -> None:
    frame = comparison_table(results, split="test", sort_by="f1_macro")
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert list(frame["model"]) == ["best", "mediocre", "worst"]
    assert frame.loc[0, "accuracy"] == pytest.approx(1.0)
    assert frame.loc[0, "n_train"] == 100


def test_comparison_table_missing_auc_becomes_nan_and_sorts_last(
    results: list[BenchmarkResult],
) -> None:
    frame = comparison_table(results, split="test", sort_by="roc_auc")
    assert np.isnan(frame.loc[len(frame) - 1, "roc_auc"])
    assert frame.loc[len(frame) - 1, "model"] == "worst"


def test_comparison_table_sorts_wall_seconds_ascending(results: list[BenchmarkResult]) -> None:
    frame = comparison_table(results, sort_by="wall_seconds")
    assert list(frame["wall_seconds"]) == [1.0, 5.0, 60.0]


def test_comparison_table_uses_requested_split(results: list[BenchmarkResult]) -> None:
    frame = comparison_table(results, split="validation")
    assert len(frame) == 3  # every result carries a validation report too


def test_comparison_table_empty_results() -> None:
    frame = comparison_table([])
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert frame.empty


def test_missing_split_raises(results: list[BenchmarkResult]) -> None:
    partial = _result("test-only", 1, seconds=2.0, splits=("test",))
    with pytest.raises(ConfigurationError, match="validation"):
        comparison_table([*results, partial], split="validation")


def test_unknown_sort_column_raises(results: list[BenchmarkResult]) -> None:
    with pytest.raises(ConfigurationError, match="sort"):
        comparison_table(results, sort_by="vibes")


def test_to_markdown_table(results: list[BenchmarkResult]) -> None:
    markdown = to_markdown_table(results)
    lines = markdown.splitlines()
    assert lines[0].startswith("| Model | Accuracy | F1 (macro) |")
    assert lines[2].startswith("| best |")  # sorted best-first by f1_macro
    assert "| n/a |" in markdown  # the proba-less model has no ROC AUC
    assert len(lines) == 2 + len(results)


def test_save_load_round_trip(results: list[BenchmarkResult], tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "benchmark.json"
    save_benchmark(results, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert load_benchmark(path) == results


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"schema_version": 999, "results": []}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="schema version"):
        load_benchmark(path)


def test_load_rejects_non_benchmark_json(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="benchmark"):
        load_benchmark(path)


def test_report_for_missing_split_raises(results: list[BenchmarkResult]) -> None:
    with pytest.raises(ConfigurationError, match="no report for split"):
        results[0].report_for("dev")
