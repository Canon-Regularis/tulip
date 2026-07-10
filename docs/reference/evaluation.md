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

::: tulip.evaluation.dataset_card

::: tulip.evaluation.model_card
