# tulip

**Polish Dialect Typology and Regional Speech Classification Analysis System.**

tulip detects Polish dialects from written text, transcribed speech, and raw
audio. It ships classical ML baselines and deep-learning models behind one API,
explains every prediction, draws it on a map of Poland, and builds reproducible,
speaker-disjoint benchmark splits — because there is currently no widely adopted
benchmark for Polish dialect identification.

!!! tip "Zero-acquisition start"
    You do not need to download anything to try tulip. The `synthetic` corpus is
    generated in-process, so a fresh clone runs end to end:

    ```bash
    tulip train configs/synthetic_text.yaml
    ```

    See the [Quickstart](guide/quickstart.md). It is a **benchmark fixture, not
    real speech** — its scores say nothing about real-world dialect ID.

## What it does

- **Text classification** — char/word TF-IDF, stylometry, affix frequencies, a
  dialect-keyword lexicon, and sub-lexical phonological markers, into Naive
  Bayes, logistic regression, calibrated linear SVM, random forest, XGBoost, or
  LightGBM; or fine-tuned HerBERT / Polish RoBERTa / mBERT / XLM-R and fastText.
- **Speech classification** — MFCC, mel spectrogram, pitch, formants, energy,
  ZCR, spectral centroid, chroma, and wav2vec2 embeddings; or wav2vec2 / HuBERT
  / Whisper fine-tuning and ECAPA-TDNN / x-vector embedding models.
- **Explainability** — signed TF-IDF evidence, LIME, SHAP, transformer attention
  maps, and nearest training examples.
- **Visualisation** — interactive folium maps highlighting the predicted region,
  confidence heatmaps, probability charts, and t-SNE/UMAP embedding spaces.
- **Reproducible benchmarks** — frozen speaker-disjoint splits, audited build
  manifests, and multi-model comparison tables from identical data.
- **Uncertainty** — every prediction carries the full probability distribution,
  top-k, and optional abstention below a confidence threshold.

## Where to go next

- **[Quickstart](guide/quickstart.md)** — the zero-download path: train, predict,
  and serve a model in three commands.
- **[Synthetic corpora](guide/synthetic-corpora.md)** — what the built-in
  fixtures are, and why their scores are not real accuracy.
- **[Serving](guide/serving.md)** — the FastAPI inference service, its endpoints,
  and the Prometheus metrics surface.
- **[API Reference](reference/index.md)** — auto-generated docs for the public
  Python API: pipeline, data, features, and evaluation.

## Design in one paragraph

Everything is registry-driven: datasets, feature extractors, models, and
explainers register under canonical names, and experiments reference them by name
plus params in YAML, so adding a component never touches core code. Feature
extractors follow the scikit-learn `fit`/`transform` contract and classifiers
follow `fit`/`predict`/`predict_proba`, so classical and deep components are
interchangeable in a pipeline. Heavy dependencies (torch, transformers, librosa,
…) are imported lazily, so `import tulip` stays cheap. The full contract lives in
the [architecture reference](architecture.md).

## Project status

Alpha. The toolkit, tests, and benchmark machinery are complete, and the
`synthetic` corpus makes them runnable end to end with no downloads. The **real**
corpora still require local acquisition — see [Datasets](datasets.md) — and
trained-model quality depends on the data you assemble. Contributions welcome;
see the [development guide](development.md).
