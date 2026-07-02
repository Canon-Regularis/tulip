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

The binding architecture contract — module layout, canonical registry names,
and the conventions reviews enforce (lazy optional imports, seeded
randomness, UTF-8 IO, `TulipError` subclasses) — is
[docs/architecture.md](architecture.md). Read it before adding a subsystem.

## Adding components

Everything extends through registries; no core code changes.

**A model** — register a factory in `tulip/models/` (see
`tulip/models/classical.py`):

```python
from tulip.models.registry import MODELS

@MODELS.register("my_model")
def my_model(*, alpha: float = 1.0, **params):
    return SomeSklearnCompatibleClassifier(alpha=alpha, **params)
```

It is immediately usable as `model: {name: my_model, params: {alpha: 0.5}}`
in experiment YAML, in `tulip benchmark -m my_model`, and in
`DialectClassifier(model="my_model")`. Contract: `fit`/`predict`/
`predict_proba`/`classes_`; heavy dependencies imported lazily via
`tulip.utils.optional.optional_import` inside methods, never at module level.

**A feature extractor** — register an sklearn transformer in
`TEXT_FEATURES` or `AUDIO_FEATURES` (`tulip/features/registries.py`).

**A dataset** — subclass `ManifestBackedLoader`
(`tulip/data/loaders/_base.py`), set `dataset_name` and `label_defaults`,
register with `@DATASETS.register(...)`, add a `DatasetInfo` entry to
`tulip/data/catalog.py`, and document the local layout in
[docs/datasets.md](datasets.md).

**An explainer** — register in `EXPLAINERS` (`tulip/explain/registry.py`);
implement `explain(pipeline, raw_input, **kwargs) -> Explanation`.

Each addition ships with tests under `tests/` (flat, area-prefixed file
names, e.g. `test_models_*.py`); optional-dependency tests guard with
`pytest.importorskip`.

## Testing philosophy

- The synthetic corpus in `tests/conftest.py` (`make_samples`) covers three
  dialects plus standard Polish with multiple speakers per class — enough to
  exercise stratified speaker-disjoint splitting and end-to-end training.
- Anything pure-Python is tested exactly (hand-computed metrics, split
  disjointness, dedup determinism).
- Heavy-model paths (torch, speechbrain, fasttext) are covered by
  construction/registration tests everywhere, and `slow`-marked smoke tests
  where a GPU-less run is realistic.

## Release checklist

1. `python -m pytest` green locally and in CI; `ruff format --check`,
   `ruff check`, `mypy` clean.
2. Bump `version` in `pyproject.toml`.
3. Update README/docs for any new components or CLI surface.
4. Tag and build: `python -m pip install build && python -m build`.
