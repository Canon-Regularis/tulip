"""High-level pipeline: the DialectClassifier facade and experiment runners."""

from tulip.pipeline.active import (
    STRATEGIES,
    AcquisitionCandidate,
    AcquisitionContext,
    AcquisitionStrategy,
    rank_for_labeling,
)
from tulip.pipeline.active_loop import (
    ActiveLoopPoint,
    ActiveLoopReport,
    active_learning_loop,
)
from tulip.pipeline.calibrated import CalibratedClassifier
from tulip.pipeline.classifier import DialectClassifier, LabelledBatch
from tulip.pipeline.conformal import (
    ConformalClassifier,
    ConformalPrediction,
    ConformalReport,
)
from tulip.pipeline.crossval import (
    CVConfig,
    CVFoldResult,
    CVReport,
    MetricSummary,
    grouped_stratified_kfold,
    run_cross_validation,
)
from tulip.pipeline.distill import DistillationConfig, DistillationReport, distill
from tulip.pipeline.experiment import (
    ExperimentResult,
    collect_predictions,
    evaluate_samples,
    run_benchmark,
    run_experiment,
)
from tulip.pipeline.fusion import (
    FusionStrategy,
    LogarithmicPoolingFusion,
    MaximumFusion,
    MultimodalClassifier,
    ProbabilisticClassifier,
    WeightedAverageFusion,
)
from tulip.pipeline.hierarchical import (
    AllOf,
    AlwaysAccept,
    AnyOf,
    BackoffPolicy,
    ConfidenceThreshold,
    HierarchicalConfig,
    HierarchicalDialectClassifier,
    MarginThreshold,
    NotAbstained,
)
from tulip.pipeline.isogloss_diagnostics import (
    IsoglossDiagnostic,
    IsoglossReport,
    isogloss_diagnostics,
)
from tulip.pipeline.learning_curve import (
    LearningCurvePoint,
    LearningCurveReport,
    learning_curve,
)
from tulip.pipeline.openset import OpenSetClassifier, OpenSetPrediction, OpenSetReport
from tulip.pipeline.protocols import SamplePredictor
from tulip.pipeline.selftrain import SelfTrainConfig, SelfTrainResult, self_train

__all__ = [
    "STRATEGIES",
    "AcquisitionCandidate",
    "AcquisitionContext",
    "AcquisitionStrategy",
    "ActiveLoopPoint",
    "ActiveLoopReport",
    "AllOf",
    "AlwaysAccept",
    "AnyOf",
    "BackoffPolicy",
    "CVConfig",
    "CVFoldResult",
    "CVReport",
    "CalibratedClassifier",
    "ConfidenceThreshold",
    "ConformalClassifier",
    "ConformalPrediction",
    "ConformalReport",
    "DialectClassifier",
    "DistillationConfig",
    "DistillationReport",
    "ExperimentResult",
    "FusionStrategy",
    "HierarchicalConfig",
    "HierarchicalDialectClassifier",
    "IsoglossDiagnostic",
    "IsoglossReport",
    "LabelledBatch",
    "LearningCurvePoint",
    "LearningCurveReport",
    "LogarithmicPoolingFusion",
    "MarginThreshold",
    "MaximumFusion",
    "MetricSummary",
    "MultimodalClassifier",
    "NotAbstained",
    "OpenSetClassifier",
    "OpenSetPrediction",
    "OpenSetReport",
    "ProbabilisticClassifier",
    "SamplePredictor",
    "SelfTrainConfig",
    "SelfTrainResult",
    "WeightedAverageFusion",
    "active_learning_loop",
    "collect_predictions",
    "distill",
    "evaluate_samples",
    "grouped_stratified_kfold",
    "isogloss_diagnostics",
    "learning_curve",
    "rank_for_labeling",
    "run_benchmark",
    "run_cross_validation",
    "run_experiment",
    "self_train",
]
