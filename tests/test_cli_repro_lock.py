"""Tests for `tulip repro verify-lock` (reproducibility without redistribution)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tulip.cli.app import app

runner = CliRunner()

_CONFIG = Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "char_baseline.yaml"


def _build_splits(seed: int | None = None):
    from tulip.config import load_experiment_config
    from tulip.data import DatasetBuilder

    config = load_experiment_config(_CONFIG)
    if seed is not None:
        config = config.model_copy(update={"split": config.split.model_copy(update={"seed": seed})})
    return config, DatasetBuilder(config.data).build(config.split, target=config.target)


def test_verify_lock_passes_on_reproduction(tmp_path: Path) -> None:
    from tulip.data.fingerprint import fingerprint_splits

    _, splits = _build_splits()
    lock = tmp_path / "split_lock.json"
    fingerprint_splits(splits).save(lock)

    result = runner.invoke(app, ["repro", "verify-lock", str(_CONFIG), str(lock)])
    assert result.exit_code == 0, result.output
    assert "reproduce the committed lock" in result.output


def test_verify_lock_fails_on_drift(tmp_path: Path) -> None:
    # A lock from a differently-seeded split must not verify against the default
    # split: the fingerprints (and sizes) differ, so verify_splits raises.
    from tulip.core.exceptions import DataError
    from tulip.data.fingerprint import SplitFingerprint, fingerprint_splits, verify_splits

    _, splits_default = _build_splits()
    _, splits_reseeded = _build_splits(seed=999)
    lock = tmp_path / "reseeded_lock.json"
    fingerprint_splits(splits_reseeded).save(lock)

    with pytest.raises(DataError):
        verify_splits(splits_default, SplitFingerprint.load(lock))

    # ...and the CLI surfaces it as a clean non-zero exit, not a traceback.
    result = runner.invoke(app, ["repro", "verify-lock", str(_CONFIG), str(lock)])
    assert result.exit_code != 0
