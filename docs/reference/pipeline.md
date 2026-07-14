# Pipeline

The entry point. The `DialectClassifier` facade, the composed classifiers
(hierarchical, calibrated, multimodal), and the experiment and benchmark runners.
These are the objects most users touch.

## Classifiers

::: tulip.pipeline.DialectClassifier

::: tulip.pipeline.HierarchicalDialectClassifier

::: tulip.pipeline.CalibratedClassifier

::: tulip.pipeline.MultimodalClassifier

::: tulip.pipeline.SamplePredictor

## Experiment and benchmark runners

::: tulip.pipeline.run_experiment

::: tulip.pipeline.run_benchmark

::: tulip.pipeline.evaluate_samples

::: tulip.pipeline.ExperimentResult

## Semi-supervised self-training

::: tulip.pipeline.self_train

::: tulip.pipeline.SelfTrainConfig
