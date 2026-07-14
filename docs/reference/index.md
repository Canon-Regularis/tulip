# API Reference

These pages are generated from the source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/). They always match the code in the
working tree. Each page shows the public objects a subsystem re-exports from its
package `__init__`.

## The subsystems

- **[Pipeline](pipeline.md).** The entry point. The `DialectClassifier` facade,
  the hierarchical, calibrated, and multimodal classifiers, and the experiment
  and benchmark runners.
- **[Data](data.md).** The dataset catalog, the `DatasetBuilder`, manifest and
  validation helpers, and the synthetic-corpus generator.
- **[Features](features.md).** The text feature extractors and the
  `build_text_features` composer.
- **[Evaluation](evaluation.md).** Metrics and reports, confusion matrices,
  calibration, benchmarks, leaderboards, significance, and cards.
- **[Deployment and serving](deploy.md).** The model registry and the serving
  settings.

## How the pieces relate

A typical experiment flows through the subsystems in order. **Data** builds a
leakage-free split. **Features** turn raw strings into vectors. **Pipeline**
composes features and a model into a `DialectClassifier` and trains it.
**Evaluation** scores the result into a report. Every component is referenced by
name in YAML. The full contract is in the
[architecture document](../architecture.md).

## Conventions

- Feature extractors implement `fit`/`transform`. Classifiers implement
  `fit`/`predict`/`predict_proba` and expose `classes_`.
- Loaders emit `Sample`. Classifiers emit `Prediction`. Explainers emit
  `Explanation`. No subsystem invents a parallel record type.
- Heavy dependencies load lazily. Importing any of these packages is cheap and
  never requires an optional extra.
