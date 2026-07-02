"""Comparison layer of the reproducible public benchmark.

Orchestration (training several models on identical frozen splits) lives in
``tulip.pipeline``; this module defines what a benchmark *result* looks like
and how a set of results becomes a comparison table, a markdown snippet for
READMEs, and a versioned JSON artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation._format import format_metric, markdown_table
from tulip.evaluation.report import EvaluationReport
from tulip.utils.io import read_json, write_json
from tulip.utils.logging import get_logger

logger = get_logger(__name__)

BENCHMARK_SCHEMA_VERSION = 1
"""Schema version written by :func:`save_benchmark`; bump on breaking changes."""

_TABLE_COLUMNS = (
    "model",
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "n_train",
    "wall_seconds",
)
# Columns where ascending order is the natural reading (lexical or lower-is-better).
_ASCENDING_COLUMNS = frozenset({"model", "wall_seconds"})


class BenchmarkResult(BaseModel):
    """One model's outcome on one benchmark experiment, across evaluated splits.

    ``reports`` maps split names (e.g. ``"validation"``, ``"test"``) to the
    :class:`EvaluationReport` computed on that split.
    """

    model_config = ConfigDict(frozen=True)

    experiment: str
    model: str
    target_level: str = "dialect"
    reports: dict[str, EvaluationReport] = Field(default_factory=dict)
    wall_seconds: float = Field(default=0.0, ge=0.0)
    n_train: int = Field(default=0, ge=0)
    n_test: int = Field(default=0, ge=0)

    def report_for(self, split: str) -> EvaluationReport:
        """Return the report for ``split``.

        Raises:
            ConfigurationError: If the split was not evaluated for this result.
        """
        try:
            return self.reports[split]
        except KeyError:
            available = sorted(self.reports)
            raise ConfigurationError(
                f"benchmark result {self.model!r} ({self.experiment!r}) has no report for "
                f"split {split!r}; available splits: {available}"
            ) from None


def comparison_table(
    results: Sequence[BenchmarkResult],
    split: str = "test",
    sort_by: str = "f1_macro",
) -> pd.DataFrame:
    """Build a one-row-per-model comparison DataFrame for a given split.

    Args:
        results: Benchmark results to compare (typically one per model over
            identical frozen splits).
        split: Which split's report to read from each result.
        sort_by: Column to order by. Metric columns sort descending (best
            first); ``model`` and ``wall_seconds`` sort ascending. ``None``
            ROC AUC values become ``NaN`` and sort last.

    Returns:
        A DataFrame with columns ``model``, ``accuracy``, ``f1_macro``,
        ``f1_weighted``, ``roc_auc``, ``n_train``, ``wall_seconds`` (empty but
        correctly shaped when ``results`` is empty).

    Raises:
        ConfigurationError: If ``sort_by`` is not a table column or any result
            lacks the requested split.
    """
    if sort_by not in _TABLE_COLUMNS:
        raise ConfigurationError(
            f"cannot sort by {sort_by!r}; expected one of {list(_TABLE_COLUMNS)}"
        )
    rows = []
    for result in results:
        report = result.report_for(split)
        auc = report.roc_auc_macro_ovr
        rows.append(
            {
                "model": result.model,
                "accuracy": report.accuracy,
                "f1_macro": report.f1_macro,
                "f1_weighted": report.f1_weighted,
                # NaN (not None) keeps the column float-typed and sortable.
                "roc_auc": float("nan") if auc is None else auc,
                "n_train": result.n_train,
                "wall_seconds": result.wall_seconds,
            }
        )
    frame = pd.DataFrame(rows, columns=list(_TABLE_COLUMNS))
    frame = frame.sort_values(
        sort_by,
        ascending=sort_by in _ASCENDING_COLUMNS,
        na_position="last",
        kind="mergesort",  # stable, so ties keep input order
    )
    return frame.reset_index(drop=True)


def to_markdown_table(
    results: Sequence[BenchmarkResult],
    split: str = "test",
    sort_by: str = "f1_macro",
) -> str:
    """Render the comparison as a markdown table for READMEs and reports.

    Args:
        results: Benchmark results to compare.
        split: Which split's report to use.
        sort_by: Ordering column, as in :func:`comparison_table`.

    Returns:
        A GitHub-flavoured markdown table; unavailable ROC AUC renders as ``n/a``.
    """
    frame = comparison_table(results, split=split, sort_by=sort_by)
    headers = ("Model", "Accuracy", "F1 (macro)", "F1 (weighted)", "ROC AUC", "Train", "Seconds")
    rows = [
        (
            str(row.model),
            format_metric(row.accuracy),
            format_metric(row.f1_macro),
            format_metric(row.f1_weighted),
            format_metric(None if pd.isna(row.roc_auc) else float(row.roc_auc)),
            str(int(row.n_train)),
            format_metric(float(row.wall_seconds), digits=1),
        )
        for row in frame.itertuples(index=False)
    ]
    return markdown_table(headers, rows)


def save_benchmark(results: Sequence[BenchmarkResult], path: Path | str) -> None:
    """Write benchmark results to ``path`` as versioned UTF-8 JSON.

    Args:
        results: The results to persist.
        path: Target JSON file; parent directories are created.
    """
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "results": [result.model_dump(mode="json") for result in results],
    }
    write_json(Path(path), payload)
    logger.debug("saved %d benchmark result(s) to %s", len(results), path)


def load_benchmark(path: Path | str) -> list[BenchmarkResult]:
    """Read benchmark results previously written by :func:`save_benchmark`.

    Args:
        path: JSON file produced by :func:`save_benchmark`.

    Returns:
        The reconstructed results, in saved order.

    Raises:
        ConfigurationError: If the file is not a benchmark artifact or its
            schema version is unsupported.
    """
    data: Any = read_json(Path(path))
    if not isinstance(data, dict) or "schema_version" not in data or "results" not in data:
        raise ConfigurationError(
            f"{path} is not a tulip benchmark file (expected 'schema_version' and 'results')"
        )
    version = data["schema_version"]
    if version != BENCHMARK_SCHEMA_VERSION:
        raise ConfigurationError(
            f"unsupported benchmark schema version {version!r} in {path}; "
            f"this tulip version reads version {BENCHMARK_SCHEMA_VERSION}"
        )
    return [BenchmarkResult.model_validate(record) for record in data["results"]]
