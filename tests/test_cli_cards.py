"""CLI smoke tests for the card commands (dataset and model cards)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from conftest import make_manifest_experiment_config, write_manifest_corpus
from tulip.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def prepared(tmp_path: Path):
    """Train a model and build splits, returning (model_dir, build_manifest_path)."""
    from tulip.config import load_experiment_config, save_experiment_config
    from tulip.data import BUILD_MANIFEST_NAME, DatasetBuilder
    from tulip.pipeline import DialectClassifier

    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="card-cli")
    save_experiment_config(config, tmp_path / "config.yaml")
    config = load_experiment_config(tmp_path / "config.yaml")

    splits_dir = tmp_path / "splits"
    splits = DatasetBuilder(config.data).build(
        config.split, target=config.target, output_dir=splits_dir
    )
    model_dir = tmp_path / "model"
    DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42).fit(
        splits.train
    ).save(model_dir)
    return model_dir, splits_dir / BUILD_MANIFEST_NAME


def test_card_model(prepared) -> None:
    model_dir, _ = prepared
    result = runner.invoke(app, ["card", "model", str(model_dir)])
    assert result.exit_code == 0, result.output
    assert "logistic_regression" in result.output


def test_card_datasheet(tmp_path: Path) -> None:
    from tulip.config import load_experiment_config, save_experiment_config
    from tulip.data import DatasetBuilder

    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="ds-cli")
    save_experiment_config(config, tmp_path / "config.yaml")
    config = load_experiment_config(tmp_path / "config.yaml")
    splits_dir = tmp_path / "splits"
    DatasetBuilder(config.data).build(config.split, target=config.target, output_dir=splits_dir)

    spec = tmp_path / "spec.yaml"
    spec.write_text("motivation: benchmark study\nuses: research only\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["card", "datasheet", str(splits_dir), "--spec", str(spec), "--dataset", "dialektarium"],
    )
    assert result.exit_code == 0, result.output
    assert "# Datasheet: dialektarium" in result.output
    assert "## Composition" in result.output
    assert "## Geographic distribution" in result.output


def test_card_benchmark(tmp_path: Path) -> None:
    board = tmp_path / "board"
    board.mkdir()
    (board / "leaderboard.md").write_text(
        "| Experiment | Model | F1 (macro) |\n| :--- | ---: | ---: |\n| real | majority | 0.1 |\n",
        encoding="utf-8",
    )
    out = tmp_path / "benchmark.md"
    result = runner.invoke(
        app, ["card", "benchmark", str(board), "--synthetic", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    doc = out.read_text(encoding="utf-8")
    assert "## Label hierarchy" in doc and "## Protocol" in doc and "## Results" in doc
    assert "Synthetic fixture" in doc


def test_card_model_to_file(prepared, tmp_path: Path) -> None:
    model_dir, _ = prepared
    out = tmp_path / "model_card.md"
    result = runner.invoke(app, ["card", "model", str(model_dir), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file() and out.read_text(encoding="utf-8").strip()


def test_card_dataset(prepared) -> None:
    _, build_manifest = prepared
    assert build_manifest.is_file(), "data build should have written a build manifest"
    result = runner.invoke(app, ["card", "dataset", str(build_manifest)])
    assert result.exit_code == 0, result.output
    assert result.output.strip()
