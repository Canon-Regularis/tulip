"""High-level pipeline: the DialectClassifier facade and experiment runners."""

from tulip.pipeline.calibrated import CalibratedClassifier
from tulip.pipeline.classifier import DialectClassifier, LabelledBatch
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
    "CalibratedClassifier",
    "ConfidenceThreshold",
    "DialectClassifier",
    "ExperimentResult",
    "FusionStrategy",
    "HierarchicalConfig",
    "HierarchicalDialectClassifier",
    "LabelledBatch",
    "LogarithmicPoolingFusion",
    "MarginThreshold",
    "MaximumFusion",
    "MultimodalClassifier",
    "NotAbstained",
    "ProbabilisticClassifier",
    "SamplePredictor",
    "SelfTrainConfig",
    "SelfTrainResult",
    "WeightedAverageFusion",
    "evaluate_samples",
    "run_benchmark",
    "run_experiment",
    "self_train",
]
