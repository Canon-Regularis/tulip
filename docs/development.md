# Development guide

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"           # plus any extras you work on, e.g. .[viz,explain]
pre-commit install
```

## Everyday commands

```bash
python -m pytest -q                 # full suite (optional-dep tests skip cleanly)
python -m pytest -q -m "not slow"   # what CI runs
python -m ruff format src tests     # format
python -m ruff check src tests      # lint (--fix for autofixes)
python -m mypy                      # type check (config in pyproject.toml)
```

The architecture contract is in [docs/architecture.md](architecture.md). It
covers the module layout, the canonical registry names, and the conventions
reviews enforce (lazy optional imports, seeded randomness, UTF-8 IO, `TulipError`
subclasses). Read it before adding a subsystem.

## Adding components

Everything extends through registries. No core code changes.

**A model.** Register a factory in `tulip/models/` (see
`tulip/models/classical.py`):

```python
from tulip.models.registry import MODELS

@MODELS.register("my_model")
def my_model(*, alpha: float = 1.0, **params):
    return SomeSklearnCompatibleClassifier(alpha=alpha, **params)
```

It is then usable as `model: {name: my_model, params: {alpha: 0.5}}` in an
experiment YAML, in `tulip benchmark -m my_model`, and in
`DialectClassifier(model="my_model")`. The contract is
`fit`/`predict`/`predict_proba`/`classes_`. Import heavy dependencies lazily with
`tulip.utils.optional.optional_import` inside methods, never at module level.

**A feature extractor.** Register an sklearn transformer in `TEXT_FEATURES` or
`AUDIO_FEATURES` (`tulip/features/registries.py`).

**A dataset.** Subclass `ManifestBackedLoader` (`tulip/data/loaders/_base.py`).
Set `dataset_name`, `label_defaults`, and an `acquisition` string (shown by
`tulip data download`). Register with `@DATASETS.register(...)`. Add a
`DatasetInfo` entry to `tulip/data/catalog.py`. Document the local layout in
[docs/datasets.md](datasets.md). If the corpus has a licence-clean bulk source,
set `auto_downloadable = True` and override `download(root, **options)` to write
the documented layout (see `BigosLoader`).

**An explainer.** Register in `EXPLAINERS` (`tulip/explain/registry.py`).
Implement `explain(pipeline, raw_input, **kwargs) -> Explanation`.

Each addition ships with tests under `tests/`. File names are flat and
area-prefixed, e.g. `test_models_*.py`. Optional-dependency tests guard with
`pytest.importorskip`.

## Testing philosophy

- The synthetic corpus in `tests/conftest.py` (`make_samples`) covers three
  dialects plus standard Polish, with several speakers per class. That is enough
  to exercise stratified speaker-disjoint splitting and end-to-end training.
- Pure-Python code is tested exactly: hand-computed metrics, split disjointness,
  dedup determinism.
- Heavy-model paths (torch, speechbrain, fasttext) get construction and
  registration tests everywhere. `slow`-marked smoke tests cover the paths where
  a GPU-less run is realistic.

## Release checklist

1. `python -m pytest` green locally and in CI. `ruff format --check`,
   `ruff check`, and `mypy` clean.
2. Bump `version` in `pyproject.toml`.
3. Update the README and docs for any new components or CLI surface.
4. Tag and build: `python -m pip install build && python -m build`.
