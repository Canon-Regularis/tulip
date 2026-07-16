# tulip

Polish Dialect Typology and Regional Speech Classification Analysis System

**tulip** detects Polish dialects. It works on written text, transcribed speech,
and raw audio. It gives you classical and deep-learning models behind one API. It
explains each prediction and can draw it on a map of Poland. It also builds
reproducible, speaker-disjoint benchmark splits. There is no widely adopted
benchmark for Polish dialect identification, so tulip provides one.

## What it does

- **Text classification.** TF-IDF, stylometry, affixes, a dialect-keyword
  lexicon, and phonological features. Classical models (Naive Bayes, logistic
  regression, SVM, random forest, boosting) and transformers (HerBERT, RoBERTa,
  mBERT, XLM-R, fastText).
- **Speech classification.** MFCC, mel spectrogram, pitch, formants, and wav2vec2
  features. Neural models (wav2vec2, HuBERT, Whisper, ECAPA-TDNN, x-vectors).
- **Explainability.** TF-IDF evidence, LIME, SHAP, attention maps, nearest
  examples, and named dialect phenomena.
- **Visualisation.** Interactive maps, confidence heatmaps, probability charts,
  and embedding plots.
- **Reproducible benchmarks.** Frozen speaker-disjoint splits, audited manifests,
  and a committed leaderboard.
- **Uncertainty.** Every prediction carries the full probability distribution,
  top-k, and optional abstention.

## Install

You need Python 3.11 or newer. The core install is light. Heavy stacks are
opt-in extras.

From PyPI the package is published as `tulip-dialect`; it still imports as
`tulip` and installs the `tulip` CLI:

```bash
pip install tulip-dialect
```

From a clone:

```bash
pip install -e .                  # core: sklearn pipelines, CLI, benchmarks
pip install -e ".[boosting]"      # XGBoost + LightGBM baselines
pip install -e ".[transformers]"  # HerBERT / RoBERTa / mBERT / XLM-R (torch)
pip install -e ".[audio]"         # librosa/parselmouth audio features
pip install -e ".[speech]"        # wav2vec2 / HuBERT / Whisper / speechbrain
pip install -e ".[explain]"       # SHAP + LIME
pip install -e ".[viz]"           # folium maps, plotly/matplotlib charts
pip install -e ".[serve]"         # FastAPI inference service
pip install -e ".[dev]"           # tests, ruff, mypy, pre-commit
```

## Quickstart

No data is needed. The `synthetic` corpus is generated in memory, so a fresh
clone runs end to end.

```bash
tulip train configs/synthetic_text.yaml      # generate, split, train, evaluate, persist
tulip train configs/synthetic_audio.yaml     # the same, end to end on synthesised audio
tulip leaderboard benchmarks/suite.yaml      # regenerate the reproducible leaderboard
tulip serve artifacts/synthetic-text/model   # HTTP API + demo UI at http://127.0.0.1:8000/
```

Both are test fixtures, not real speech (see
[docs/datasets.md](docs/datasets.md)). For real corpora, which you acquire
locally because tulip never scrapes at runtime:

```bash
tulip data list                              # catalog, tiers, local availability
tulip data download --all                    # fetch automatic sources; print manual steps
tulip data validate data/raw/dgp/manifest.csv  # check a manifest before trusting it
tulip data prepare configs/text_baseline.yaml  # build speaker-disjoint splits
tulip train configs/text_baseline.yaml       # train + evaluate + persist

tulip predict artifacts/text-baseline/model \
  "Hej, baca się pyto, kaj się owce pasą na holi." \
  --explain top_tfidf --map prediction.html

tulip benchmark configs/text_baseline.yaml \
  -m naive_bayes -m logistic_regression -m linear_svm

tulip serve artifacts/text-baseline/model    # POST /predict/text, /predict/audio
```

Or from Python:

```python
from tulip import DialectClassifier
from tulip.data import DatasetBuilder
from tulip.config import load_experiment_config

config = load_experiment_config("configs/text_baseline.yaml")
splits = DatasetBuilder(config.data).build(config.split, target=config.target)

clf = DialectClassifier(model="logistic_regression",
                        features=["char_tfidf", "word_tfidf", "stylometry"])
clf.fit(splits.train)

text = "Jo żech je z Katowic i godom po naszymu."
prediction = clf.predict(text)
print(prediction.label, prediction.top_k(3))          # silesia, ...
print(clf.explain(text, method="top_tfidf").top_attributions(5))
```

## Label taxonomy

Labels are hierarchical: village, region, regional dialect, dialect family, plus
voivodeship. Models can train at any level.

| Family | Regional dialects (gwary) |
| --- | --- |
| Greater Polish (wielkopolski) | Greater Poland, Kujawy, Kociewie |
| Lesser Polish (małopolski) | Lesser Poland, Podhale, Spisz, Orawa, Podolia |
| Masovian (mazowiecki) | Mazovia, Kurpie, Warmia, Masuria, Podlasie |
| Silesian (śląski) | Silesia, Cieszyn Silesia |
| Kashubian | Kashubia |
| Standard Polish | none (negative class for dialect-vs-standard) |

## Architecture

tulip is registry-driven. Datasets, features, models, and explainers register
under names. Experiments reference them by name in YAML. Adding a component does
not touch core code. The full contract is in
[docs/architecture.md](docs/architecture.md).

```text
src/tulip/
  core/        types (Sample, Prediction, Explanation), Registry, interfaces
  labels/      dialect taxonomy + region/voivodeship geography
  data/        catalog, loaders, cleaning, dedup, speaker-disjoint splitting
  features/    text/ and audio/ extractor registries
  models/      classical, transformer text, neural speech, persistence
  evaluation/  metrics, reports, confusion matrices, benchmark tables
  explain/     top_tfidf, lime, shap, attention, nearest_examples
  viz/         prediction maps, confidence heatmaps, charts, embedding space
  pipeline/    DialectClassifier facade + experiment/benchmark runners
  cli/  serve/ typer CLI and FastAPI service
  deploy/      content-addressed model registry
```

## Datasets

| Corpus | Tier | Provides | Source |
| --- | --- | --- | --- |
| Dialektarium | 1 | audio + transcriptions, village/region labels | <https://dialektarium.pl/> |
| Dialekty i gwary polskie (DGP) | 1 | dialect texts, dialect/region labels | <https://przewodnik.tmjp.pl/dgp-dialekty-i-gwary-polskie-kompendium-internetowe/> |
| Korpus Spiski | 2 | Spisz dialect transcriptions | <https://journals.akademicka.pl/lv/article/view/727> |
| Maćkowce corpus (EKTGM) | 2 | Podolia borderland dialect texts | <https://przewodnik.tmjp.pl/ektgm-elektroniczny-korpus-tekstow-gwarowych-z-mackowiec-na-podolu/> |
| NKJP | 3 | standard Polish (negatives) | <https://nkjp.pl/> |
| Spokes | 3 | conversational spoken Polish | <https://spokes.clarin-pl.eu/> |
| Common Voice PL | 3 | read speech + accent metadata | <https://commonvoice.mozilla.org/> |
| BIGOS | 4 | aggregated Polish ASR corpora | <https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos> |
| `synthetic` | n/a | generated dialect text (fixture, not real speech) | generated in memory |
| `synthetic_audio` | n/a | generated dialect audio (fixture, not real speech) | generated in memory |

Acquisition steps, the manifest format, and the synthetic corpora are in
[docs/datasets.md](docs/datasets.md). Full API docs build with `mkdocs serve`
(install the `docs` extra).

## Project status

Alpha. The toolkit, tests, and benchmark machinery are complete. The `synthetic`
corpus runs them end to end with no downloads, but it is a fixture, so its scores
say nothing about real accuracy. The real corpora need local acquisition. Most
sources have no bulk download and unclear redistribution rights. Model quality
depends on the data you assemble. There are no published benchmark numbers on
real dialect data yet. Contributions are welcome (see
[docs/development.md](docs/development.md)).

## License

MIT. See [LICENSE](LICENSE).
