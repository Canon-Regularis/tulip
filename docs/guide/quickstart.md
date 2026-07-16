# Quickstart

This guide takes a fresh clone to a trained model, a prediction, and a running
service. It needs no data. The `synthetic` corpus is generated in memory, so
every command below works today.

!!! warning "The synthetic corpus is a fixture, not real speech"
    These numbers show that the machinery works. They are not real dialect
    accuracy. See [Synthetic corpora](synthetic-corpora.md).

## Install

You need Python 3.11 or newer. The core install is light. Heavy stacks are
opt-in extras. From PyPI the package is published as `tulip-dialect`; it still
imports as `tulip` and installs the `tulip` CLI:

```bash
pip install tulip-dialect
```

For development from a clone:

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"           # core + tooling
```

Add extras only for the stacks you use. For example: `.[boosting]`,
`.[transformers]`, `.[audio]`, `.[speech]`, `.[explain]`, `.[viz]`, `.[serve]`.

Run `tulip doctor` to see which models, features, and explainers run on your
install, and the exact `pip install` for anything still blocked.

## 1. Train

Train a text pipeline on the synthetic corpus. One command does it all. It
generates the corpus, cleans and deduplicates it, builds a speaker-disjoint
split, trains, evaluates on validation and test, and saves the model.

```bash
tulip train configs/synthetic_text.yaml
```

The artifact lands under the experiment's `output_dir`. Next to it sit the
resolved config, `build_manifest.json`, and per-split reports.

!!! note "Audio parity"
    Audio works the same way. `tulip train configs/synthetic_audio.yaml` trains
    on synthetic audio features with the same flow. Everything below applies to
    audio models too.

To train at the family level (which adds the `standard` class), use
`configs/synthetic_family.yaml`.

## 2. Predict

Point `tulip predict` at the saved model directory and pass some text. Add
`--explain` for evidence. Add `--map` to render a map of Poland.

```bash
tulip predict artifacts/synthetic-text/model \
  "Jo żech je z Katowic i godom po naszymu." \
  --explain top_tfidf --map prediction.html
```

Use `--json` for machine-readable output. Use `--audio path/to/clip.wav` with an
audio model instead of text.

### From Python

The same flow is available as a library. `DialectClassifier` composes features
and a model into one object.

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

Expose the saved model over HTTP. This needs the `serve` extra.

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

The service also serves a browser demo at `/`, a `/health` probe, batch and
audio endpoints, and Prometheus `/metrics`. See [Serving](serving.md).

## 4. Benchmark and leaderboard

Compare several models on one frozen split. Then regenerate the committed,
byte-stable leaderboard.

```bash
tulip benchmark configs/synthetic_text.yaml \
  -m naive_bayes -m logistic_regression -m linear_svm
tulip leaderboard benchmarks/suite.yaml
```

Deduplication runs before splitting. Grouping keeps speakers disjoint. Results
are comparable across runs. The details are in
[Datasets](../datasets.md#building-a-reproducible-benchmark-split).

## Working with real corpora

Real dialect corpora are acquired locally. tulip never scrapes at runtime. The
usual loop:

1. `tulip data download --all` fetches the licence-clean sources and prints
   manual steps for the rest.
2. Assemble a manifest.
3. `tulip data validate` checks it.
4. `tulip data prepare`, then `tulip train`.

See [Datasets](../datasets.md) for acquisition, the manifest format, and
per-corpus layouts.
