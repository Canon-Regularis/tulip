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

## Active learning

Rank an unlabeled pool by which samples to label first, so a fixed annotation
budget buys the most signal. Strategies are a registry, each owning its own
parameters: the uncertainty measures `least_confidence`, `margin`, and
`entropy`, plus the dialect-aware `intensity_gated`, which multiplies uncertainty
by dialect intensity so budget is not spent on standard Polish. Ranking is a pure
function of the model and pool, ties broken by sample id.

::: tulip.pipeline.rank_for_labeling

::: tulip.pipeline.AcquisitionCandidate

::: tulip.pipeline.STRATEGIES

::: tulip.pipeline.AcquisitionStrategy

## Cross-validation

Grouped, stratified K-fold cross-validation. Folds are speaker-disjoint. Each
metric gets a mean and a 95% confidence interval across all folds and seeds.

::: tulip.pipeline.run_cross_validation

::: tulip.pipeline.CVConfig

::: tulip.pipeline.CVReport

::: tulip.pipeline.grouped_stratified_kfold

## Conformal prediction

Distribution-free prediction sets. The set covers the true label at least
`1 - alpha` of the time. `mondrian=True` gives per-class coverage.

::: tulip.pipeline.ConformalClassifier

::: tulip.pipeline.ConformalPrediction

::: tulip.pipeline.ConformalReport

## Open-set novelty

Flag inputs unlike any known dialect. A row whose every class is excluded from
the conformal set conforms to no known dialect and is novel. A test sample whose
gold dialect was never trained on is the ground truth for evaluation.

::: tulip.pipeline.OpenSetClassifier

::: tulip.pipeline.OpenSetPrediction

::: tulip.pipeline.OpenSetReport
