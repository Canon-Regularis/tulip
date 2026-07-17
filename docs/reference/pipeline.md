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

## Multimodal fusion

`MultimodalClassifier` fuses a text and an audio expert at the probability level
with a pluggable strategy. Beyond the weighted, maximum, and log-pool strategies,
`ConfidenceWeightedFusion` applies a per-sample soft attention over the two
experts: each is weighted by its own confidence on that sample, so the more
certain modality carries more weight where it is reliable. It is parameter-free
and data-driven, a lightweight stand-in for a learned multimodal transformer.

`compare_modalities` answers the question a single leaderboard row cannot: does
fusion actually help? It scores text-only, audio-only, and their fusion on one
identical multimodal test set and reports the fusion uplift with a paired McNemar
test against each single modality. It is exposed as
`tulip fusion-compare TEXT_MODEL AUDIO_MODEL DATA --strategy <name>`.

::: tulip.pipeline.fusion.ConfidenceWeightedFusion

::: tulip.pipeline.fusion.compare_modalities

::: tulip.pipeline.fusion.ModalityComparison

## Experiment and benchmark runners

::: tulip.pipeline.run_experiment

::: tulip.pipeline.run_benchmark

::: tulip.pipeline.evaluate_samples

::: tulip.pipeline.ExperimentResult

## Learning curves

How metric quality scales with training-set size. The model trains on nested,
stratified fractions of the training split and scores each on the identical
held-out test split, so the curve isolates training-set size as the only moving
part. Deterministic and byte-stable, so a saved report re-runs identically.

::: tulip.pipeline.learning_curve

::: tulip.pipeline.LearningCurveReport

::: tulip.pipeline.LearningCurvePoint

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

### Closed-loop active learning

Simulate acquire, label, and retrain over the training split (its gold labels are
the oracle), scoring the held-out test split each round. Run it against
`strategy="random"` to see whether a strategy beats labeling at random.

::: tulip.pipeline.active_learning_loop

::: tulip.pipeline.ActiveLoopReport

::: tulip.pipeline.ActiveLoopPoint

## Distillation

Distil a large teacher into a small, fast student: the teacher labels a transfer
pool, the student trains on those labels, and the report puts the student's
accuracy retention next to the size and latency it costs.

::: tulip.pipeline.distill

::: tulip.pipeline.DistillationReport

::: tulip.pipeline.DistillationConfig

## Isogloss diagnostics

Ask whether accuracy collapses when a dialect marker is absent. For each
detectable isogloss, the samples of the dialects it signals are split by whether
the reflex fired in the text and accuracy is compared across the split.

::: tulip.pipeline.isogloss_diagnostics

::: tulip.pipeline.IsoglossReport

::: tulip.pipeline.IsoglossDiagnostic

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
