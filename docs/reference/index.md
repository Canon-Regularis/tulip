# API Reference

These pages are generated directly from the source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/), so they always match the code in
the working tree. Each page renders the genuinely public objects re-exported from
a subsystem's package `__init__` (its `__all__`).

## The four subsystems

- **[Pipeline](pipeline.md)** — the user-facing entry point. The
  `DialectClassifier` facade plus the hierarchical, calibrated, and multimodal
  classifiers, and the experiment/benchmark runners.
- **[Data](data.md)** — the dataset catalog, the `DatasetBuilder` (load → clean →
  dedup → speaker-disjoint split → persist), manifest/validation helpers, and the
  synthetic-corpus generator.
- **[Features](features.md)** — the text feature extractors (TF-IDF, stylometry,
  affixes, dialect keywords, phonological markers) and the `build_text_features`
  composer.
- **[Evaluation](evaluation.md)** — metrics and reports, confusion matrices,
  calibration, the benchmark comparison schema, leaderboards, and cards.

## How the pieces relate

A typical experiment flows top to bottom through these subsystems: **data**
builds a leakage-free split, **features** turn raw strings into vectors,
**pipeline** composes features and a model into a `DialectClassifier` and trains
it, and **evaluation** scores the result into a reproducible report. The
registry-driven design means every component is referenced by a canonical string
name in YAML — the full contract is in the
[architecture document](../architecture.md).

## Conventions

- Feature extractors implement the scikit-learn `fit`/`transform` contract;
  classifiers implement `fit`/`predict`/`predict_proba` and expose `classes_`.
- Loaders emit `tulip.core.types.Sample`; classifiers emit `Prediction`;
  explainers emit `Explanation`. No subsystem invents a parallel record type.
- Heavy dependencies are imported lazily, so importing any of these packages is
  cheap and never requires an optional extra to be installed.
