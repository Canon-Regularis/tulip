"""CLI smoke tests for the leaderboard and efficiency commands."""

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
def suite_path(tmp_path: Path) -> Path:
    """A tiny one-config leaderboard suite over a manifest corpus."""
    from tulip.config import save_experiment_config

    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="board-cli")
    config_path = tmp_path / "board.yaml"
    save_experiment_config(config, config_path)

    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: cli-board\n"
        f"configs:\n  - {config_path.as_posix()}\n"
        "models:\n  - naive_bayes\n  - logistic_regression\n",
        encoding="utf-8",
    )
    return suite


def test_leaderboard_writes_a_board(suite_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    result = runner.invoke(app, ["leaderboard", str(suite_path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    board = out / "cli-board" / "leaderboard.md"
    assert board.is_file()
    text = board.read_text(encoding="utf-8")
    assert "naive_bayes" in text and "logistic_regression" in text


def test_leaderboard_parallel_matches_sequential(suite_path: Path, tmp_path: Path) -> None:
    # --jobs runs models in separate processes; the committed-style board is
    # byte-identical to the sequential run.
    seq = tmp_path / "seq"
    par = tmp_path / "par"
    assert runner.invoke(app, ["leaderboard", str(suite_path), "--out", str(seq)]).exit_code == 0
    result = runner.invoke(app, ["leaderboard", str(suite_path), "--out", str(par), "--jobs", "2"])
    assert result.exit_code == 0, result.output
    seq_board = (seq / "cli-board" / "leaderboard.md").read_bytes()
    par_board = (par / "cli-board" / "leaderboard.md").read_bytes()
    assert seq_board == par_board


def test_efficiency_command(tmp_path: Path) -> None:
    from tulip.data import DatasetBuilder, save_splits
    from tulip.pipeline import DialectClassifier

    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="eff-cli")
    splits = DatasetBuilder(config.data).build(config.split, target=config.target)
    model_dir = tmp_path / "model"
    DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42).fit(
        splits.train
    ).save(model_dir)
    paths = save_splits(splits, tmp_path / "splits")

    result = runner.invoke(app, ["efficiency", str(model_dir), str(paths["test"]), "--json"])
    assert result.exit_code == 0, result.output
    import json

    record = json.loads(result.output)
    assert record["latency_ms"] >= 0.0
