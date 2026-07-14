# tulip architecture

**tulip** (Polish Dialect Typology and Regional Speech Classification Analysis System)
detects Polish dialects from written text, transcribed speech, and raw audio. This
document is the binding contract between subsystems: module layout, canonical
component names, shared conventions, and public API expectations.

## Design principles

1. **Registry-driven modularity.** Datasets, feature extractors, models, and
   explainers register themselves under canonical string names in a
   `tulip.core.registry.Registry`. Experiment configs reference components by
   name + params. Adding a component never requires touching core code.
   Components declare capabilities as registration `metadata` (e.g. models
   whose constructors accept the shared TrainingConfig knobs register with
   `metadata={"training_aware": True}`); consumers query `Registry.metadata`
   instead of hardcoding per-component knowledge.
2. **scikit-learn conventions as the lingua franca.** Feature extractors
   implement `fit`/`transform`; classifiers implement `fit`/`predict`/
   `predict_proba` and expose `classes_`. Neural models wrap themselves in this
   API so classical and deep components are interchangeable in pipelines.
3. **Lazy heavy imports.** Core install stays light. torch, transformers,
   librosa, speechbrain, shap, lime, folium, plotly, matplotlib, fasttext,
   xgboost, lightgbm, datasets MUST be imported only inside functions/methods
   via `tulip.utils.optional.optional_import(module, extra=..., purpose=...)`.
   Module import and registration must never require an optional dependency.
4. **Canonical data types.** Loaders produce `tulip.core.types.Sample`;
   classifiers produce `Prediction`; explainers produce `Explanation`. No
   subsystem invents parallel record types.
5. **Reproducibility.** Splits are speaker-disjoint and seeded; experiments are
   fully declared in YAML; artifacts are saved with their config and metrics.
   This is what makes tulip usable as a public benchmark.
6. **No runtime scraping.** Dataset loaders read documented local directory
   layouts under `data/raw/<dataset>/`; acquisition is documented per corpus in
   `docs/datasets.md`. Hugging Face-hosted corpora may use the `datasets`
   library (optional extra `hf`).

## Package layout and ownership

```text
src/tulip/
  core/          # types, registry, interfaces, exceptions        [frozen]
  labels/        # taxonomy (families, dialects), geo centroids   [frozen]
  utils/         # optional imports, io, logging, seeding         [frozen]
  config/        # pydantic experiment schemas + YAML loader      [frozen]
  data/          # dataset catalog, loaders, cleaning, dedup, splitting, builder,
                 # split fingerprint (reproducibility)
  features/
    registries.py  # TEXT_FEATURES / AUDIO_FEATURES registries    [frozen]
    text/          # char/word n-grams, stylometry, affixes, keyword lexicon
    audio/         # MFCC, mel, pitch, formants, ZCR, centroid, chroma, wav2vec2
  models/
    registry.py    # MODELS registry                              [frozen]
    _common.py     # shared estimator machinery (label encoding, fit
                   # validation, seed reconciliation, argmax-predict mixin,
                   # torch training + batched-softmax inference loops)
    classical.py   # NB, LogReg, LinearSVM, RF, XGBoost, LightGBM
    neural_text.py # HerBERT, Polish RoBERTa, mBERT, XLM-R fine-tuning
    neural_audio.py# wav2vec2, HuBERT, Whisper-encoder, ECAPA-TDNN, x-vectors
    fasttext_model.py
    persistence.py # save/load trained pipelines with metadata
  evaluation/    # metrics, EvaluationReport, confusion, calibration, leaderboard,
                 # significance/selective/error-analysis, dataset/model cards
  explain/       # top TF-IDF, LIME, SHAP, attention, nearest examples
  viz/           # folium region map + confidence heatmap, charts, embedding space
  pipeline/      # DialectClassifier facade + experiment runner
  cli/           # typer app (entry point: tulip.cli.app:main)
  serve/         # FastAPI single interface for text + audio upload
```

Files marked `[frozen]` are foundation-owned: do not modify them. If a frozen
API blocks you, code around it and report the friction in your summary instead
of editing it.

## Canonical component names

Registry names are lowercase snake_case. These names are load-bearing: configs,
docs, and tests refer to them.

- `tulip.data.DATASETS` (`Registry` defined in `tulip/data/registry.py`):
  `dialektarium`, `dgp`, `korpus_spiski`, `mackowce`, `nkjp`, `spokes`,
  `common_voice_pl`, `bigos`, `manifest` (generic CSV/JSONL manifest loader),
  `synthetic` (generated in-process; needs no acquisition, so the toolkit runs
  end-to-end on a fresh clone — see `docs/datasets.md`),
  `synthetic_audio` (the audio analogue: writes deterministic 16 kHz WAV clips
  whose per-class pitch/formants/spectral tilt make the classical audio features
  separable, so the audio path is exercised end-to-end too).
- `tulip.features.TEXT_FEATURES`:
  `char_tfidf`, `word_tfidf`, `stylometry`, `affix_frequency`, `dialect_keywords`,
  `phonological_markers` (sub-lexical isoglosses the whole-word lexicon cannot
  encode: soft-labial clusters, and the sibilant-digraph rate that makes
  mazurzenie legible as a conspicuous *absence*).
- `tulip.features.AUDIO_FEATURES`:
  `mfcc`, `mel_spectrogram`, `pitch`, `formants`, `energy`, `zero_crossing_rate`,
  `spectral_centroid`, `chroma`, `wav2vec2_embeddings`.
- `tulip.models.MODELS`:
  `naive_bayes`, `logistic_regression`, `linear_svm`, `random_forest`,
  `xgboost`, `lightgbm`, `herbert`, `polish_roberta`, `mbert`, `xlm_roberta`,
  `fasttext`, `wav2vec2`, `hubert`, `whisper`, `ecapa_tdnn`, `xvector`.
- `tulip.explain.EXPLAINERS`:
  `top_tfidf`, `lime`, `shap`, `attention`, `nearest_examples`.

## Module contracts

### tulip.data

- `registry.py`: `DATASETS: Registry` holding `DatasetLoader` **classes**.
- `catalog.py`: declarative `DatasetInfo` metadata for every corpus (tiers 1-4,
  URLs, tasks, contents, label levels) + `catalog()` accessor.
- `manifest.py`: the generic manifest reader (`read_manifest`,
  `ManifestColumns`, `surrogate_speaker_id`) every manifest-backed loader
  delegates to.
- `reading.py`: `read_samples(path)` — labelled samples back from anything
  tulip writes or documents (split JSONL, manifest file, manifest directory);
  shared by `tulip evaluate` and library callers.
- `download.py`: `download_datasets(names, root)` — fetches corpora whose
  loader is `auto_downloadable` (loaders override `DatasetLoader.download`)
  and returns `MANUAL` reports carrying each remaining corpus's
  `acquisition` steps; surfaced as `tulip data download`.
- Loaders subclass `tulip.core.interfaces.DatasetLoader` (`info` property,
  `load(root) -> Iterator[Sample]`). Loaders are generous in what they accept
  (CSV/TSV/JSONL manifests) and strict in what they emit (validated `Sample`s
  with `speaker_id` filled — synthesise a stable surrogate from available
  metadata when the corpus lacks explicit speaker IDs).
- `cleaning.py`: composable text normalisation (`TextCleaner`): unicode NFC,
  whitespace collapse, quote/dash normalisation, transcription-artifact removal
  (e.g. `[śmiech]`, `...`, annotation markup), optional lowercasing. Preserve
  dialectal orthography — never strip diacritics or "correct" spelling.
- `dedup.py`: exact dedup on normalised-text hash + near-duplicate detection
  via character-shingle Jaccard similarity. Pure stdlib/numpy.
- `splitting.py`: `speaker_disjoint_split(samples, config: SplitConfig) ->
  DatasetSplits` (named train/validation/test lists). Guarantees zero speaker
  overlap across splits (group-aware), attempts stratification by the
  configured label level, deterministic under a fixed seed. Must raise
  `DataError` when a split would be empty.
- `builder.py`: `DatasetBuilder` orchestrating load -> clean -> dedup -> split
  -> persist (JSONL per split + a manifest with counts and config hash).

### tulip.features.text

Extractors are sklearn transformers (subclass `TransformerMixin, BaseEstimator`
where sensible) operating on sequences of strings:

- `char_tfidf` / `word_tfidf`: thin, well-defaulted wrappers over
  `TfidfVectorizer` (char_wb 2-5 grams; word 1-2 grams).
- `stylometry`: dense features — sentence length stats, word length stats,
  punctuation frequencies, type-token ratio, hapax ratio, uppercase ratio.
  Expose `get_feature_names_out()`.
- `affix_frequency`: frequencies of word-initial prefixes and word-final
  suffixes (configurable lengths), hashed or vocabulary-based.
- `dialect_keywords`: lexicon-based counts of known dialect marker words; ship
  a starter lexicon (well-attested markers per dialect, e.g. Podhale "ka/kaj",
  archaic aorist "-ch", Silesian "godać", mazurzenie respellings) as package
  data with provenance comments; lexicon must be user-extensible via a path
  param.
- `build_text_features(configs: list[ComponentConfig]) -> FeatureUnion` helper.

### tulip.features.audio

Extractors take sequences of audio file paths and return fixed-size row
vectors (frame-level features pooled with mean+std by a shared `pooling.py`).
librosa/soundfile/parselmouth imported lazily; formants fall back to an
LPC-based estimate (scipy) when parselmouth is unavailable. A shared
`loading.py` handles decoding + resampling to 16 kHz mono.

### tulip.models

- `classical.py`: registered factories returning sklearn-compatible
  classifiers. `linear_svm` must wrap `LinearSVC` in
  `CalibratedClassifierCV` so `predict_proba` exists. `xgboost`/`lightgbm`
  guard their imports and encode string labels internally.
- `neural_text.py`: `TransformerTextClassifier` (sklearn-style wrapper around
  HF `AutoModelForSequenceClassification`): registered names map to
  checkpoints — herbert `allegro/herbert-base-cased`, polish_roberta
  `sdadas/polish-roberta-base-v2`, mbert `bert-base-multilingual-cased`,
  xlm_roberta `xlm-roberta-base`. Accepts raw texts in `fit(X, y)`.
- `neural_audio.py`: `SpeechClassifier` wrappers — wav2vec2
  `facebook/wav2vec2-xls-r-300m`, hubert, whisper encoder; `ecapa_tdnn` /
  `xvector` via speechbrain embeddings + a light classification head. Accept
  audio paths in `fit(X, y)`.
- `persistence.py`: `save_model(pipeline, path, metadata)` /
  `load_model(path)` using joblib, storing a JSON sidecar (tulip version,
  config, classes, metrics).

### tulip.evaluation

- `metrics.py`: `compute_metrics(y_true, y_pred, y_proba=None, labels=None) ->
  EvaluationReport` — accuracy, macro/weighted precision/recall/F1, per-class
  breakdown, macro one-vs-rest ROC AUC (guarded: omit when y_proba is absent
  or a class is missing), confusion matrix.
- `report.py`: `EvaluationReport` pydantic model with `to_markdown()`,
  `save(path)`.
- `benchmark.py`: the benchmark result schema (`BenchmarkResult`), comparison
  tables, and JSON persistence — the reporting half of the
  reproducible-benchmark deliverable. The orchestration half (`run_benchmark`,
  training several models over identical frozen splits) lives in
  `tulip.pipeline.experiment`, which layers *above* evaluation.

### tulip.explain

`EXPLAINERS` registry; implementations satisfy `tulip.core.interfaces.Explainer`
and return `tulip.core.types.Explanation`. `top_tfidf` reads linear-model
coefficients through a fitted sklearn Pipeline; `nearest_examples` retrieves
cosine-similar training samples; `lime`/`shap`/`attention` guard their imports.

### tulip.viz

- `map.py`: folium map builders — `prediction_map(prediction)` highlighting the
  predicted region (top-3 shown with graded opacity) and
  `confidence_heatmap(prediction)` shading all regions by probability, using
  `tulip.labels.geo` centroids. Return the folium `Map`; provide `save(path)`.
- `charts.py`: probability bar chart, confusion-matrix heatmap (matplotlib or
  plotly, lazy).
- `embedding_space.py`: 2-D projection (t-SNE core, UMAP optional) of sample or
  dialect embeddings for cluster visualisation.

### tulip.pipeline

- `classifier.py`: `DialectClassifier` — the user-facing facade. Composes
  feature configs + model config into one trainable object (`task`, `target`,
  and `abstain_threshold` are constructor arguments); `fit(samples)`,
  `predict(raw) -> Prediction` (top-k probabilities, abstention below
  `abstain_threshold`), `predict_batch`, `predict_proba`,
  `labelled_batch(samples) -> LabelledBatch` (the public raw-input/label
  pairing used by training and evaluation), `explain(raw, method=...)`
  (routing in `explaining.py`), `save`/`load` via `models.persistence`.
- `experiment.py`: `run_experiment(config: ExperimentConfig) -> ExperimentResult`
  — seed, load+prepare data, split, train, evaluate on validation+test, persist
  model, metrics, splits, and the resolved config under
  `output_dir/<experiment-name>/`. Also `evaluate_samples(classifier, samples)`,
  `collect_predictions(classifier, samples) -> SplitPredictions` (the per-sample
  substrate, sharing one inference pass), and `run_benchmark(config, models)`
  (several models, one frozen split). All three accept an optional
  `calibration_bins` to populate the report's ECE/MCE/Brier block.

### tulip.cli (typer)

`tulip.cli.app:main` — command groups: `data`
(list/download/prepare/synthesize/synthesize-audio/validate), `train`,
`evaluate`, `predict` (text arg or `--audio` path; `--json`; map export via
`--map out.html`; inline explanations via `--explain <method>`), `explain`
(standalone; the command group the contract lists), `benchmark`, `leaderboard`
(also emits significance), `analyze` (selective + error report from a saved
`predictions_<split>.json`), `repro verify` (regenerate a suite and fail on
drift from the committed board), `card` (dataset/model), `selftrain`, `serve`.
Rich tables for human output; `--json` for machine output. `data validate` and
`repro verify` exit non-zero on failure so they can gate CI.

### tulip.evaluation (benchmark surface)

- `leaderboard.py`: `LeaderboardSuite` + `run_leaderboard`/`write_leaderboard`
  over the untouched `run_benchmark`. `leaderboard.md` and `provenance.json` are
  deterministic — no timestamps, no `wall_seconds` — so a committed leaderboard
  regenerates byte-identically. Rows are keyed by `(experiment, model)`. The
  board carries ECE/Brier columns when the suite sets `calibration_bins`, and
  `write_significance` emits per-experiment paired-significance artifacts.
- `_provenance_env.py`: the deterministic `environment` block for provenance —
  Python floor + key dependency versions read from the committed `uv.lock` (not
  the live interpreter) + content digests of the configs and lexicons.
- `predictions.py`: `SplitPredictions` / `PredictionRecord` — the per-sample
  substrate (gold, prediction, probability row, self-describing slice keys) the
  three rigor analyses below share. Built by
  `tulip.pipeline.experiment.collect_predictions`.
- `significance.py`: `paired_significance` — bootstrap CIs per metric, exact
  Holm-corrected McNemar between models on the identical paired split, and a
  "tied with best" grouping. SciPy-free (`math.comb`), seeded, deterministic.
- `selective.py`: `selective_report` — risk-coverage curve, AURC, accuracy at a
  target coverage, coverage at a target error, over the abstention the
  classifier already ships.
- `error_analysis.py`: `error_report` — most-confused pairs, hard exemplars, and
  per-slice (source/speaker/length/modality) fairness metrics.
- `cards.py`: `dataset_card` / `model_card` render byte-stable markdown from
  artifacts the toolkit already writes (`build_manifest.json`, `metadata.json`,
  `report_<split>.json`).

### tulip.pipeline (semi-supervised)

- `selftrain.py`: `self_train` grows a classifier from a labelled seed set using
  confident pseudo-labels, so label-less corpora (e.g. `bigos`) contribute.
  Knobs live in a module-owned `SelfTrainConfig` — `ExperimentConfig` is frozen
  and forbids extra fields.

### tulip.pipeline (classifier composition)

`protocols.py` defines `SamplePredictor` — one method, `predict_samples(samples)
-> list[Prediction]`. It exists because the classifiers below **must not**
subclass `DialectClassifier`: `predict_batch` guarantees every `Prediction` has
`level == self.target` over a single modality, and each of these violates one of
those postconditions. Relating them by protocol rather than inheritance is the
Liskov substitution principle being *obeyed*, not sidestepped. `DialectClassifier`
satisfies the protocol via a `predict_samples` adapter.

- `hierarchical/`: `HierarchicalDialectClassifier` composes one
  `DialectClassifier` per `LabelLevel` (coarse → fine) and returns the finest
  prediction a `BackoffPolicy` accepts, so `Prediction.level` varies per sample.
  With `mask_to_coarse`, a dialect row is projected onto the predicted family by
  the chain rule — rescaled to `P(family) · P(dialect | family)`, *not*
  renormalised to 1 — so a child can never out-confidence its parent, and a
  family with no dialects (`standard`) forces a backoff instead of a guess.
  Policies (`ConfidenceThreshold`, `MarginThreshold`, `NotAbstained`,
  `AlwaysAccept`, `AllOf`/`AnyOf`) are frozen value objects behind a one-method
  protocol.
- `calibrated.py`: `CalibratedClassifier` wraps any classifier with a
  `ProbabilityCalibrator` fitted on a **held-out** split, and applies
  `abstain_threshold` to the *calibrated* top probability — an uncalibrated
  cutoff does not mean what it looks like.
- `fusion/`: `MultimodalClassifier` fuses a text and an audio classifier via a
  `FusionStrategy` (weighted average, maximum, logarithmic pooling), aligning
  their classes to the sorted union and degrading to whichever modality a sample
  actually carries. `TaskType` is frozen, so this is composition rather than a
  `MULTIMODAL` enum member. (Both `hierarchical/` and `fusion/` are packages: a
  leaf `policies`/`strategies` module plus the classifier, so the value-object
  families are testable without the classifier stack.)

### tulip.evaluation (calibration)

- `calibration.py`: `compute_calibration` returns a `CalibrationReport` with
  top-label ECE, MCE, and the multiclass Brier score (range `[0, 2]`).
  `EvaluationReport.calibration` is opt-in via `compute_metrics(...,
  calibration_bins=N)` so existing artifacts stay byte-identical.
- `tulip.models.calibration`: `TemperatureScaling` (on `log p` as surrogate
  logits — softmax is invariant to the additive constant), `IsotonicCalibrator`,
  and `IdentityCalibrator` as a Null Object.

### tulip.serve (FastAPI)

`create_app(model_path) -> FastAPI`. Endpoints: `POST /predict/text` (JSON
body), `POST /predict/text/batch` (a list of texts), `POST /predict/audio`
(multipart upload) — all returning pydantic-native `Prediction` JSON (with the
`abstained` flag) plus `X-Tulip-Version`/`X-Model-Target`/`X-Model-Classes`
headers; `GET /health` (model identity, class count, abstention config);
`GET /metrics` (Prometheus text exposition, dependency-free); and `GET /`, a
self-contained demo UI (inline SVG Poland map + probability bars). One HTTP
middleware assigns/echoes an `X-Request-ID`, times each request
(`X-Process-Time-Ms`), records the metrics, and emits one structured log line
per request.

## Conventions (enforced)

- Python >= 3.10; `from __future__ import annotations` in every module; full
  type hints on public APIs; PEP 604 unions (`str | None`).
- Ruff: line length 100, rule set in `pyproject.toml`. Run
  `ruff format` + `ruff check` on your files before finishing.
- Google-style docstrings on every public module/class/function; explain
  *why* where non-obvious, not narration of the code.
- Logging via `tulip.utils.logging.get_logger(__name__)`; never `print` in
  library code (CLI/serve output via rich/typer is fine).
- Errors: raise subclasses of `TulipError` (`DataError`,
  `ConfigurationError`, `MissingDependencyError` via `optional_import`).
- Randomness always flows from an explicit seed; use
  `numpy.random.default_rng(seed)`, never module-level global state.
- Paths are `pathlib.Path`; IO is UTF-8; keep everything Windows-safe (no
  `/tmp`, no POSIX-only calls, no filenames differing only by case).
- Tests: pytest, files flat under `tests/` with area-prefixed unique names
  (`test_data_*.py`, `test_features_text_*.py`, ...). Tests requiring optional
  deps guard with `pytest.importorskip`. Every module ships tests for its
  pure-Python logic; heavy-model tests are construction/config tests plus
  `slow`-marked smoke tests.
