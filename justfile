# Task runner for tulip. `just` is entirely optional -- every recipe below is a
# single command you can copy and run by hand, and CI does not depend on it.
# Its only job is to keep local commands and .github/workflows/ci.yml identical,
# so "it passed locally" means something.
#
#   just            # list recipes
#   just check      # everything CI gates on
#
# Install: https://github.com/casey/just

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# The project is developed on Windows and tested on Linux in CI.
py := if os_family() == "windows" { ".venv\\Scripts\\python.exe" } else { ".venv/bin/python" }

# The extra set CI installs. torch-backed extras (transformers/speech) are
# omitted deliberately: their tests skip via pytest.importorskip.
extras := "dev,boosting,viz,explain,serve,hf,audio"

_default:
    @just --list

# Install the package and the CI extra set into ./.venv
install:
    {{ py }} -m pip install -e ".[{{ extras }}]"

# Format sources in place
fmt:
    {{ py }} -m ruff format src tests

# Fail if anything is unformatted (what CI runs)
fmt-check:
    {{ py }} -m ruff format --check src tests

# Lint
lint:
    {{ py }} -m ruff check src tests

# Type-check (blocking in CI)
typecheck:
    {{ py }} -m mypy

# Full test suite, including slow-marked tests
test:
    {{ py }} -m pytest -q

# Skip slow-marked tests (what CI runs)
test-fast:
    {{ py }} -m pytest -q -m "not slow"

# Tests with coverage; fails below the fail_under floor in pyproject.toml
cov:
    {{ py }} -m pytest -q -m "not slow" --cov=tulip --cov-report=term

# Re-resolve uv.lock after changing dependencies in pyproject.toml
lock:
    {{ py }} -m uv lock

# Fail if uv.lock is stale relative to pyproject.toml (what CI checks)
lock-check:
    {{ py }} -m uv lock --check

# Everything CI gates on, in CI's order
check: fmt-check lint typecheck lock-check cov

# Write the generated synthetic corpus to data/raw/synthetic
synth:
    {{ py }} -m tulip.cli.app data synthesize --out data/raw/synthetic

# Regenerate the committed, byte-reproducible leaderboard
bench:
    {{ py }} -m tulip.cli.app leaderboard benchmarks/suite.yaml --out benchmarks/results

# Build a wheel and smoke-test it (what the CI `build` job does)
build:
    {{ py }} -m pip install --quiet build twine
    {{ py }} -m build
    {{ py }} -m twine check dist/*
