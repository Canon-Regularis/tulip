# Evaluation

Metrics, reports, confusion matrices, calibration, and the reproducible-benchmark
surface. The typical flow: `compute_metrics` turns predictions into a frozen
`EvaluationReport`; `BenchmarkResult` collects per-split reports for one model so
several models can be compared and persisted; leaderboards and cards render
byte-stable markdown from committed artifacts.

## Metrics and reports

::: tulip.evaluation.compute_metrics

::: tulip.evaluation.EvaluationReport

::: tulip.evaluation.ClassMetrics

## Confusion matrices

::: tulip.evaluation.confusion_from_report

::: tulip.evaluation.plot_confusion

## Calibration

::: tulip.evaluation.compute_calibration

::: tulip.evaluation.CalibrationReport

## Benchmarks

::: tulip.evaluation.BenchmarkResult

::: tulip.evaluation.comparison_table

::: tulip.evaluation.to_markdown_table

::: tulip.evaluation.save_benchmark

::: tulip.evaluation.load_benchmark

## Leaderboards and cards

::: tulip.evaluation.LeaderboardSuite

::: tulip.evaluation.run_leaderboard

::: tulip.evaluation.write_leaderboard

::: tulip.evaluation.write_significance

::: tulip.evaluation.dataset_card

::: tulip.evaluation.model_card

## Per-sample predictions

The substrate the rigor analyses below share: `collect_predictions` (in the
pipeline) turns a fitted classifier into a `SplitPredictions`, whose records
carry each sample's gold label, prediction, probability row, and slice keys.

::: tulip.evaluation.SplitPredictions

::: tulip.evaluation.PredictionRecord

## Significance

Turns a ranking into claims: bootstrap confidence intervals per metric, exact
paired McNemar tests between models on the identical frozen split, Holm
correction, and a "tied with best" grouping.

::: tulip.evaluation.paired_significance

::: tulip.evaluation.SignificanceReport

::: tulip.evaluation.mcnemar_exact

## Selective prediction

Scores the abstention trade-off the classifier already ships: a risk-coverage
curve, AURC, accuracy at a target coverage, and coverage at a target error.

::: tulip.evaluation.selective_report

::: tulip.evaluation.SelectiveReport

::: tulip.evaluation.risk_coverage_curve

## Error analysis

Derives what the confusion matrix cannot: most-confused pairs, hard exemplars,
and per-slice fairness/robustness metrics.

::: tulip.evaluation.error_report

::: tulip.evaluation.ErrorReport

::: tulip.evaluation.top_confused_pairs

::: tulip.evaluation.slice_metrics
