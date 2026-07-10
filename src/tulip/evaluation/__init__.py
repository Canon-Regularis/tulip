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
from tulip.evaluation.calibration import (
    CalibrationBin,
    CalibrationReport,
    compute_calibration,
    reliability_curve,
)
from tulip.evaluation.cards import dataset_card, dataset_card_from_splits, model_card
from tulip.evaluation.confusion import (
    NORMALIZE_OPTIONS,
    confusion_from_report,
    plot_confusion,
    to_dataframe,
)
from tulip.evaluation.leaderboard import (
    LeaderboardSuite,
    load_suite,
    render_leaderboard_markdown,
    run_leaderboard,
    write_leaderboard,
)
from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.report import ClassMetrics, EvaluationReport

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "NORMALIZE_OPTIONS",
    "BenchmarkResult",
    "CalibrationBin",
    "CalibrationReport",
    "ClassMetrics",
    "EvaluationReport",
    "LeaderboardSuite",
    "comparison_table",
    "compute_calibration",
    "compute_metrics",
    "confusion_from_report",
    "dataset_card",
    "dataset_card_from_splits",
    "load_benchmark",
    "load_suite",
    "model_card",
    "plot_confusion",
    "reliability_curve",
    "render_leaderboard_markdown",
    "run_leaderboard",
    "save_benchmark",
    "to_dataframe",
    "to_markdown_table",
    "write_leaderboard",
]
