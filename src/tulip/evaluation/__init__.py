"""Evaluation: metrics, reports, confusion matrices, and benchmark comparison.

The typical flow: :func:`compute_metrics` turns predictions into a frozen
:class:`EvaluationReport`; confusion helpers derive matrices/plots from it;
:class:`BenchmarkResult` collects per-split reports for one model so several
models can be compared with :func:`comparison_table` / :func:`to_markdown_table`
and persisted with :func:`save_benchmark` / :func:`load_benchmark`.

Importing this package never requires an optional dependency; only
:func:`plot_confusion` needs matplotlib (extra ``viz``).
"""

from __future__ import annotations

from tulip.evaluation.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkResult,
    comparison_table,
    load_benchmark,
    save_benchmark,
    to_markdown_table,
)
from tulip.evaluation.confusion import (
    NORMALIZE_OPTIONS,
    confusion_from_report,
    plot_confusion,
    to_dataframe,
)
from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.report import ClassMetrics, EvaluationReport

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "NORMALIZE_OPTIONS",
    "BenchmarkResult",
    "ClassMetrics",
    "EvaluationReport",
    "comparison_table",
    "compute_metrics",
    "confusion_from_report",
    "load_benchmark",
    "plot_confusion",
    "save_benchmark",
    "to_dataframe",
    "to_markdown_table",
]
