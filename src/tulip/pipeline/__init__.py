"""High-level pipeline: the DialectClassifier facade and experiment runners."""

from tulip.pipeline.active import (
    STRATEGIES,
    AcquisitionCandidate,
    AcquisitionContext,
    AcquisitionStrategy,
    rank_for_labeling,
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
    "ExperimentResult",
    "FusionStrategy",
    "HierarchicalConfig",
    "HierarchicalDialectClassifier",
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
    "collect_predictions",
    "evaluate_samples",
    "grouped_stratified_kfold",
    "learning_curve",
    "rank_for_labeling",
    "run_benchmark",
    "run_cross_validation",
    "run_experiment",
    "self_train",
]
