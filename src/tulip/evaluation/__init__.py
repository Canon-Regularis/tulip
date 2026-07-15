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
from tulip.evaluation.cross_corpus import (
    CrossCorpusReport,
    LocoResult,
    TransferMatrix,
    partition_by_source,
    run_loco,
    transfer_matrix,
)
from tulip.evaluation.efficiency import (
    EfficiencyRecord,
    count_parameters,
    measure_efficiency,
    model_size_bytes,
    write_efficiency,
)
from tulip.evaluation.error_analysis import (
    ConfusedPair,
    ErrorReport,
    Exemplar,
    SliceMetric,
    error_report,
    slice_metrics,
    top_confused_pairs,
)
from tulip.evaluation.fairness import (
    DisparityMetric,
    FairnessReport,
    fairness_report,
    worst_group_gap,
)
from tulip.evaluation.hierarchical_metrics import HierarchicalReport, hierarchical_metrics
from tulip.evaluation.leaderboard import (
    LeaderboardSuite,
    load_suite,
    render_leaderboard_markdown,
    run_leaderboard,
    write_leaderboard,
    write_significance,
)
from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.power import PowerReport, minimum_detectable_effect
from tulip.evaluation.predictions import PredictionRecord, SplitPredictions
from tulip.evaluation.report import ClassMetrics, EvaluationReport
from tulip.evaluation.selective import (
    SelectivePoint,
    SelectiveReport,
    risk_coverage_curve,
    selective_report,
)
from tulip.evaluation.significance import (
    MetricCI,
    ModelSignificance,
    PairwiseTest,
    SignificanceReport,
    mcnemar_exact,
    paired_significance,
)
from tulip.evaluation.tracks import (
    Track,
    TrackedSuite,
    TrackResult,
    load_tracked_suite,
    render_tracked_markdown,
    run_tracked_leaderboard,
    write_tracked_leaderboard,
)
from tulip.evaluation.uncertainty import (
    UncertaintyReport,
    decompose_uncertainty,
    member_probabilities,
    uncertainty_report,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "NORMALIZE_OPTIONS",
    "BenchmarkResult",
    "CalibrationBin",
    "CalibrationReport",
    "ClassMetrics",
    "ConfusedPair",
    "CrossCorpusReport",
    "DisparityMetric",
    "EfficiencyRecord",
    "ErrorReport",
    "EvaluationReport",
    "Exemplar",
    "FairnessReport",
    "HierarchicalReport",
    "LeaderboardSuite",
    "LocoResult",
    "MetricCI",
    "ModelSignificance",
    "PairwiseTest",
    "PowerReport",
    "PredictionRecord",
    "SelectivePoint",
    "SelectiveReport",
    "SignificanceReport",
    "SliceMetric",
    "SplitPredictions",
    "Track",
    "TrackResult",
    "TrackedSuite",
    "TransferMatrix",
    "UncertaintyReport",
    "comparison_table",
    "compute_calibration",
    "compute_metrics",
    "confusion_from_report",
    "count_parameters",
    "dataset_card",
    "dataset_card_from_splits",
    "decompose_uncertainty",
    "error_report",
    "fairness_report",
    "hierarchical_metrics",
    "load_benchmark",
    "load_suite",
    "load_tracked_suite",
    "mcnemar_exact",
    "measure_efficiency",
    "member_probabilities",
    "minimum_detectable_effect",
    "model_card",
    "model_size_bytes",
    "paired_significance",
    "partition_by_source",
    "plot_confusion",
    "reliability_curve",
    "render_leaderboard_markdown",
    "render_tracked_markdown",
    "risk_coverage_curve",
    "run_leaderboard",
    "run_loco",
    "run_tracked_leaderboard",
    "save_benchmark",
    "selective_report",
    "slice_metrics",
    "to_dataframe",
    "to_markdown_table",
    "top_confused_pairs",
    "transfer_matrix",
    "uncertainty_report",
    "worst_group_gap",
    "write_efficiency",
    "write_leaderboard",
    "write_significance",
    "write_tracked_leaderboard",
]
