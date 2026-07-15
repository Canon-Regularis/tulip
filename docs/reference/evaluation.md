# Evaluation

Metrics, reports, confusion matrices, calibration, and the benchmark surface.
`compute_metrics` turns predictions into a frozen `EvaluationReport`.
`BenchmarkResult` collects per-split reports for one model, so several models can
be compared and saved. Leaderboards and cards render byte-stable markdown.

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

The rigor analyses below share this substrate. `collect_predictions` (in the
pipeline) turns a fitted classifier into a `SplitPredictions`. Each record holds
the gold label, the prediction, the probability row, and slice keys.

::: tulip.evaluation.SplitPredictions

::: tulip.evaluation.PredictionRecord

## Significance

Turns a ranking into claims. Bootstrap confidence intervals per metric. Exact
paired McNemar tests between models on the same split. Holm correction. A "tied
with best" grouping.

::: tulip.evaluation.paired_significance

::: tulip.evaluation.SignificanceReport

::: tulip.evaluation.mcnemar_exact

## Selective prediction

Scores the abstention trade-off. A risk-coverage curve, AURC, accuracy at a
target coverage, and coverage at a target error.

::: tulip.evaluation.selective_report

::: tulip.evaluation.SelectiveReport

::: tulip.evaluation.risk_coverage_curve

## Error analysis

Goes beyond the confusion matrix. Most-confused pairs, hard exemplars, and
per-slice fairness metrics.

::: tulip.evaluation.error_report

::: tulip.evaluation.ErrorReport

::: tulip.evaluation.top_confused_pairs

::: tulip.evaluation.slice_metrics

## Cross-corpus transfer

Train on some corpora, test on another. This shows whether the model learned
dialect or corpus artifacts. Leave-one-corpus-out plus the full transfer matrix,
partitioned by sample source.

::: tulip.evaluation.run_loco

::: tulip.evaluation.CrossCorpusReport

::: tulip.evaluation.transfer_matrix

::: tulip.evaluation.TransferMatrix

## Uncertainty decomposition

Split predictive uncertainty into aleatoric (data noise) and epistemic (model
doubt) from ensemble members. `decompose_uncertainty` is a pure function over a
member-probability array; `member_probabilities` extracts that array from a
fitted voting or stacking ensemble.

::: tulip.evaluation.decompose_uncertainty

::: tulip.evaluation.member_probabilities

::: tulip.evaluation.uncertainty_report

::: tulip.evaluation.UncertaintyReport
