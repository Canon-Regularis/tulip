# tulip

Polish Dialect Typology and Regional Speech Classification Analysis System

**tulip** detects Polish dialects from written text, transcribed speech, and raw
audio. It ships classical ML baselines and deep-learning models behind one
API, explains every prediction, draws it on a map of Poland, and builds
reproducible, speaker-disjoint benchmark splits - because there is currently
no widely adopted benchmark for Polish dialect identification.

## What it does

- **Text classification** - char/word TF-IDF, stylometry, affix frequencies,
  and a dialect-keyword lexicon into Naive Bayes, logistic regression,
  calibrated linear SVM, random forest, XGBoost, or LightGBM; or fine-tuned
  HerBERT / Polish RoBERTa / mBERT / XLM-R and fastText.
- **Speech classification** - MFCC, mel spectrogram, pitch, formants, energy,
  ZCR, spectral centroid, chroma, and wav2vec2 embeddings; or wav2vec2 /
  HuBERT / Whisper fine-tuning and ECAPA-TDNN / x-vector embedding models.
- **Explainability** - signed TF-IDF evidence, LIME, SHAP, transformer
  attention maps, and nearest training examples.
- **Visualisation** - interactive folium maps highlighting the predicted
  region, confidence heatmaps over all regions, probability charts, and
  t-SNE/UMAP dialect embedding spaces.
- **Reproducible benchmarks** - frozen speaker-disjoint splits (no speaker in
  both train and test), audited build manifests, and multi-model comparison
  tables from identical data.
- **Uncertainty** - every prediction carries the full probability
  distribution, top-k, and optional abstention below a confidence threshold.

## Install

Requires Python 3.10+. The core install is deliberately light; heavy stacks
are opt-in extras:

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

Corpora are acquired locally (tulip never scrapes at runtime) - see
[docs/datasets.md](docs/datasets.md) for per-corpus instructions, then:

```bash
tulip data list                              # catalog, tiers, local availability
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

Labels are hierarchical - village → region → regional dialect → dialect
family (plus voivodeship) - and models can train at any level:

| Family | Regional dialects (gwary) |
| --- | --- |
| Greater Polish (wielkopolski) | Greater Poland, Kujawy, Kociewie |
| Lesser Polish (małopolski) | Lesser Poland, Podhale, Spisz, Orawa, Podolia |
| Masovian (mazowiecki) | Mazovia, Kurpie, Warmia, Masuria, Podlasie |
| Silesian (śląski) | Silesia, Cieszyn Silesia |
| Kashubian | Kashubia |
| Standard Polish | — (negative class for dialect-vs-standard) |

## Architecture

Everything is registry-driven: datasets, feature extractors, models, and
explainers register under canonical names, and experiments reference them by
name + params in YAML. Adding a component never touches core code. The full
contract lives in [docs/architecture.md](docs/architecture.md).

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

Acquisition steps and the manifest format are documented in
[docs/datasets.md](docs/datasets.md).

## Project status

Alpha. The toolkit, tests, and benchmark machinery are complete; corpora
require local acquisition (most sources have no bulk download and unclear
redistribution rights). Trained-model quality therefore depends on the data
you assemble. Contributions welcome — see
[docs/development.md](docs/development.md).

## License

MIT — see [LICENSE](LICENSE).
