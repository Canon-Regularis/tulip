# Quickstart

This guide takes a fresh clone to a trained model, a prediction, and a running
inference service — **with no data acquisition**. The `synthetic` corpus is
generated in-process, so every command below works today.

!!! warning "The synthetic corpus is a fixture, not real speech"
    The numbers you get here measure whether the machinery works, not real
    dialect-identification accuracy. See [Synthetic corpora](synthetic-corpora.md).

## Install

Requires Python 3.10+. The core install is deliberately light; heavy stacks are
opt-in extras.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"           # core + tooling
```

Add extras only for the stacks you use, for example `pip install -e ".[boosting]"`
(XGBoost/LightGBM), `.[transformers]` (HerBERT/RoBERTa), `.[audio]`, `.[speech]`,
`.[explain]`, `.[viz]`, or `.[serve]`.

## 1. Train

Train an end-to-end text pipeline on the synthetic dialect corpus. This one
command generates the corpus, cleans and deduplicates it, builds a
speaker-disjoint split, trains, evaluates on validation and test, and persists
the model with its config and metrics.

```bash
tulip train configs/synthetic_text.yaml
```

The trained artifact lands under the experiment's `output_dir`, alongside the
resolved config, `build_manifest.json`, and per-split evaluation reports.

!!! note "Audio parity"
    An audio path mirrors this exactly — `tulip train configs/synthetic_audio.yaml`
    trains on synthetic audio features with the same generate → split → train →
    evaluate flow. Everything below applies to audio models too.

You can also train at the family level (which includes the `standard` negative
class) with `tulip train configs/synthetic_family.yaml`.

## 2. Predict

Point `tulip predict` at the saved model directory and pass text to classify.
Add `--explain` for evidence and `--map` to render an interactive map of Poland.

```bash
tulip predict artifacts/synthetic-text/model \
  "Jo żech je z Katowic i godom po naszymu." \
  --explain top_tfidf --map prediction.html
```

Use `--json` for machine-readable output, or `--audio path/to/clip.wav` (with an
audio-trained model) instead of a text argument.

### From Python

The same flow is available as a library. The `DialectClassifier` facade composes
feature configs and a model into one trainable object:

```python
from tulip import DialectClassifier
from tulip.data import DatasetBuilder
from tulip.config import load_experiment_config

config = load_experiment_config("configs/synthetic_text.yaml")
splits = DatasetBuilder(config.data).build(config.split, target=config.target)

clf = DialectClassifier(model="logistic_regression",
                        features=["char_tfidf", "word_tfidf", "stylometry"])
clf.fit(splits.train)

prediction = clf.predict("Jo żech je z Katowic i godom po naszymu.")
print(prediction.label, prediction.top_k(3))
```

See the [pipeline reference](../reference/pipeline.md) for the full API.

## 3. Serve

Expose the saved model over HTTP (needs the `serve` extra):

```bash
pip install -e ".[serve]"
tulip serve artifacts/synthetic-text/model
```

Then classify over the network:

```bash
curl -X POST localhost:8000/predict/text \
  -H "content-type: application/json" \
  -d '{"text": "Jo żech je z Katowic i godom po naszymu."}'
```

The service also exposes a browser demo UI at `/`, a `/health` probe, batch and
audio endpoints, and Prometheus `/metrics`. See [Serving](serving.md).

## 4. Benchmark and leaderboard

Evaluate several models against one frozen split, then regenerate the committed,
byte-stable leaderboard:

```bash
tulip benchmark configs/synthetic_text.yaml \
  -m naive_bayes -m logistic_regression -m linear_svm
tulip leaderboard benchmarks/suite.yaml
```

Deduplication runs before splitting and grouping guarantees speaker disjointness,
so results are comparable across runs. The details are in
[Datasets](../datasets.md#building-a-reproducible-benchmark-split).

## Working with real corpora

Real dialect corpora are acquired locally (tulip never scrapes at runtime). The
typical loop is `tulip data download --all` (fetches the licence-clean automatic
sources, prints manual steps for the rest), assemble a manifest, then
`tulip data validate`, `tulip data prepare`, and `tulip train`. See
[Datasets](../datasets.md) for acquisition, the manifest format, and per-corpus
layouts.
