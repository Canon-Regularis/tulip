"""High-level pipeline: the DialectClassifier facade and experiment runners."""

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
from tulip.pipeline.protocols import SamplePredictor
from tulip.pipeline.selftrain import SelfTrainConfig, SelfTrainResult, self_train

__all__ = [
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
    "LogarithmicPoolingFusion",
    "MarginThreshold",
    "MaximumFusion",
    "MetricSummary",
    "MultimodalClassifier",
    "NotAbstained",
    "ProbabilisticClassifier",
    "SamplePredictor",
    "SelfTrainConfig",
    "SelfTrainResult",
    "WeightedAverageFusion",
    "evaluate_samples",
    "grouped_stratified_kfold",
    "run_benchmark",
    "run_cross_validation",
    "run_experiment",
    "self_train",
]
