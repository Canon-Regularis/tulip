"""CLI smoke tests for the modeling commands (crossval, transfer, conformal)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from conftest import make_manifest_experiment_config, write_manifest_corpus
from tulip.cli.app import app
from tulip.config import save_experiment_config

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=2)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="cli-model")
    path = tmp_path / "experiment.yaml"
    save_experiment_config(config, path)
    return path


def test_crossval_command_runs(config_path: Path) -> None:
    result = runner.invoke(app, ["crossval", str(config_path), "--k", "3", "--seeds", "0,1"])
    assert result.exit_code == 0, result.output
    assert "cross-validation" in result.output
    assert "f1_macro" in result.output


def test_transfer_command_needs_multiple_sources(config_path: Path) -> None:
    # The manifest corpus is a single source, so cross-corpus transfer is a
    # clean error rather than a traceback.
    result = runner.invoke(app, ["transfer", str(config_path)])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_conformal_command_reports_coverage(config_path: Path, tmp_path: Path) -> None:
    from tulip.config import load_experiment_config
    from tulip.data import DatasetBuilder
    from tulip.pipeline import DialectClassifier

    # Train + save a model, then split off calibration/test JSONLs for the command.
    config = load_experiment_config(config_path)
    splits = DatasetBuilder(config.data).build(config.split, target=config.target)
    model_dir = tmp_path / "model"
    DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42).fit(
        splits.train
    ).save(model_dir)

    from tulip.data import save_splits

    paths = save_splits(splits, tmp_path / "splits")
    result = runner.invoke(
        app,
        [
            "conformal",
            str(model_dir),
            str(paths["validation"]),
            str(paths["test"]),
            "--alpha",
            "0.2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "coverage" in result.output
