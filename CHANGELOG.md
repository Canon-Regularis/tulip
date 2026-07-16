# Changelog

All notable changes to tulip are recorded here. The format follows Keep a
Changelog, and the project follows Semantic Versioning.

## [Unreleased]

## [0.1.0] - 2026-07-16

First public release.

### Added

- A tag-triggered release workflow that builds the package, creates the GitHub
  Release, and publishes to PyPI with trusted publishing. The distribution is
  named `tulip-dialect` (PyPI already has an unrelated `tulip`); the import
  package and the CLI stay `tulip`.
- Documentation deployed to GitHub Pages on every push to main.
- A production serving image (`Dockerfile.serve`) that bakes a small synthetic
  demo model, so `docker run` serves the interactive demo with no setup.
- `tulip registry push` to publish a registered model to the Hugging Face Hub
  together with its generated model card.
- Dialect classification from text, transcribed speech, and raw audio behind one
  `DialectClassifier` API.
- Classical baselines, gradient boosting, fastText, transformer text models, and
  neural speech models, all registered under canonical names.
- Text and audio feature extractors, including a dialect-keyword lexicon and
  phonological rules.
- Explainability: TF-IDF evidence, LIME, SHAP, attention maps, nearest examples,
  and named dialect phenomena.
- Reproducible speaker-disjoint splits, manifest validation, and dataset and
  model cards.
- A deterministic synthetic corpus so the whole pipeline runs with no data to
  acquire.
- A committed leaderboard with a reproducibility gate that regenerates it byte
  for byte.
- Ensembles, split conformal prediction, grouped cross-validation, and
  cross-corpus transfer evaluation.
- An HTTP inference service and a content-addressed model registry.
- `tulip doctor` to report which components run now and what to install, plus
  `tulip cite` for citation metadata.

[Unreleased]: https://github.com/Canon-Regularis/tulip/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Canon-Regularis/tulip/releases/tag/v0.1.0
