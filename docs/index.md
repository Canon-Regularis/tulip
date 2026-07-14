# tulip

**Polish Dialect Typology and Regional Speech Classification Analysis System.**

tulip detects Polish dialects. It works on written text, transcribed speech, and
raw audio.

It gives you classical and deep-learning models behind one API. It explains each
prediction and can draw it on a map of Poland. It also builds reproducible,
speaker-disjoint benchmark splits. There is no widely adopted benchmark for
Polish dialect identification, so tulip provides one.

!!! tip "Zero-acquisition start"
    You do not need to download anything to try tulip. The `synthetic` corpus is
    generated in memory. A fresh clone runs end to end:

    ```bash
    tulip train configs/synthetic_text.yaml
    ```

    See the [Quickstart](guide/quickstart.md). The synthetic corpus is a test
    fixture, not real speech. Its scores do not reflect real dialect accuracy.

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

## Where to go next

- **[Quickstart](guide/quickstart.md).** Train, predict, and serve a model in
  three commands, with no download.
- **[Synthetic corpora](guide/synthetic-corpora.md).** What the built-in fixtures
  are, and why their scores are not real accuracy.
- **[Serving](guide/serving.md).** The HTTP service, its endpoints, guards, and
  metrics.
- **[API Reference](reference/index.md).** Generated docs for the public Python
  API.

## Design

tulip is registry-driven. Datasets, features, models, and explainers register
under names. Experiments reference them by name in YAML. Adding a component does
not touch core code.

Feature extractors follow the scikit-learn `fit`/`transform` contract.
Classifiers follow `fit`/`predict`/`predict_proba`. Classical and deep components
are interchangeable.

Heavy dependencies load lazily, so `import tulip` stays cheap. The full contract
is in the [architecture reference](architecture.md).

## Project status

Alpha. The toolkit, tests, and benchmark machinery are complete. The `synthetic`
corpus runs them end to end with no downloads. The real corpora need local
acquisition (see [Datasets](datasets.md)). Model quality depends on the data you
assemble. Contributions are welcome (see the [development guide](development.md)).
