# Contributing to tulip

Thanks for considering a contribution!

1. Read [docs/architecture.md](docs/architecture.md) (the module contract)
   and [docs/development.md](docs/development.md) for setup, commands, and
   worked examples of adding datasets, features, models, and explainers.
2. Keep the conventions: registry-driven components, lazy optional imports
   (`tulip.utils.optional`), full type hints, Google-style docstrings,
   `TulipError` subclasses, seeded randomness, UTF-8 everywhere.
3. Every change ships with tests; optional-dependency tests must skip cleanly
   when the extra is absent.
4. Before opening a PR: `ruff format`, `ruff check`, `pytest`, and `mypy`
   must pass (`pre-commit install` automates the first two).
5. Dataset contributions must respect source licences.
