"""Tests for the closed-loop active-learning simulation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from conftest import make_manifest_experiment_config, write_manifest_corpus
from tulip.core.exceptions import ConfigurationError
from tulip.pipeline import active_learning_loop

if TYPE_CHECKING:
    from pathlib import Path

    from tulip.config.schemas import ExperimentConfig


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=8, variants=3)
    return make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="loop")


def test_curve_grows_by_the_batch_each_round(config: ExperimentConfig) -> None:
    report = active_learning_loop(config, strategy="entropy", seed_size=6, batch_size=6, rounds=3)

    assert [point.round for point in report.points] == [0, 1, 2, 3]
    sizes = [point.n_labeled for point in report.points]
    assert sizes == [6, 12, 18, 24]  # seed, then one batch per round
    assert all(0.0 <= point.f1_macro <= 1.0 for point in report.points)
    assert report.model == "logistic_regression"
    assert report.target == "dialect"
    assert report.strategy == "entropy"
    assert report.seed_size == 6 and report.batch_size == 6


def test_loop_is_deterministic(config: ExperimentConfig, tmp_path: Path) -> None:
    first = active_learning_loop(config, strategy="entropy", seed_size=6, batch_size=6, rounds=2)
    second = active_learning_loop(config, strategy="entropy", seed_size=6, batch_size=6, rounds=2)
    assert first.model_dump() == second.model_dump()

    first.save(tmp_path / "a.json")
    second.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_random_baseline_runs(config: ExperimentConfig) -> None:
    report = active_learning_loop(config, strategy="random", seed_size=6, batch_size=6, rounds=2)
    assert report.strategy == "random"
    assert [point.n_labeled for point in report.points] == [6, 12, 18]


def test_loop_stops_when_pool_is_exhausted(config: ExperimentConfig) -> None:
    # A huge round budget cannot label more than the pool holds; the curve stops
    # growing once nothing remains to acquire.
    report = active_learning_loop(config, strategy="random", seed_size=6, batch_size=30, rounds=50)
    sizes = [point.n_labeled for point in report.points]
    assert sizes == sorted(sizes)
    assert len(sizes) < 51  # stopped early, well under the 50-round budget


def test_invalid_params_raise(config: ExperimentConfig) -> None:
    with pytest.raises(ConfigurationError, match="seed_size must be"):
        active_learning_loop(config, seed_size=0)
    with pytest.raises(ConfigurationError, match="batch_size must be"):
        active_learning_loop(config, batch_size=0)
    with pytest.raises(ConfigurationError, match="rounds must be"):
        active_learning_loop(config, rounds=0)
    with pytest.raises(ConfigurationError, match="exceeds the training pool"):
        active_learning_loop(config, seed_size=100_000)


def test_unknown_strategy_fails_fast(config: ExperimentConfig) -> None:
    with pytest.raises(ConfigurationError, match="unknown acquisition strategy"):
        active_learning_loop(config, strategy="does_not_exist")


def test_markdown_names_the_curve(config: ExperimentConfig) -> None:
    report = active_learning_loop(config, strategy="margin", seed_size=6, batch_size=6, rounds=1)
    markdown = report.to_markdown()
    assert "Active learning" in markdown
    assert "strategy=margin" in markdown
    assert "F1 (macro)" in markdown


def test_cli_active_loop(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tulip.cli.app import app
    from tulip.config.loader import save_experiment_config

    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=8, variants=3)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="loop-cli")
    config_path = tmp_path / "config.yaml"
    save_experiment_config(config, config_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "active-loop",
            str(config_path),
            "--seed-size",
            "6",
            "--batch-size",
            "6",
            "--rounds",
            "2",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["points"]) == 3

    saved = runner.invoke(
        app,
        [
            "active-loop",
            str(config_path),
            "--seed-size",
            "6",
            "--batch-size",
            "6",
            "--rounds",
            "1",
            "--out",
            str(tmp_path / "loop.json"),
        ],
    )
    assert saved.exit_code == 0, saved.output
    assert (tmp_path / "loop.json").is_file()


def test_cli_rejects_unknown_strategy(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tulip.cli.app import app
    from tulip.config.loader import save_experiment_config

    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=8, variants=3)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="loop-bad")
    config_path = tmp_path / "config.yaml"
    save_experiment_config(config, config_path)

    result = CliRunner().invoke(app, ["active-loop", str(config_path), "--strategy", "nope"])
    assert result.exit_code == 1
    assert "unknown acquisition strategy" in result.output
