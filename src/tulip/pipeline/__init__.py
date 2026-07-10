"""High-level pipeline: the DialectClassifier facade and experiment runners."""

from tulip.pipeline.classifier import DialectClassifier, LabelledBatch
from tulip.pipeline.experiment import (
    ExperimentResult,
    evaluate_samples,
    run_benchmark,
    run_experiment,
)
from tulip.pipeline.selftrain import SelfTrainConfig, SelfTrainResult, self_train

__all__ = [
    "DialectClassifier",
    "ExperimentResult",
    "LabelledBatch",
    "SelfTrainConfig",
    "SelfTrainResult",
    "evaluate_samples",
    "run_benchmark",
    "run_experiment",
    "self_train",
]
