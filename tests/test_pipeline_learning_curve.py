"""Tests for the learning-curve analysis."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from conftest import make_manifest_experiment_config, write_manifest_corpus
from tulip.core.exceptions import ConfigurationError
from tulip.pipeline import learning_curve

if TYPE_CHECKING:
    from pathlib import Path

    from tulip.config.schemas import ExperimentConfig


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    return make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="curve")


def test_points_are_nested_and_increasing(config: ExperimentConfig) -> None:
    report = learning_curve(config, fractions=(0.25, 0.5, 1.0))

    assert [point.fraction for point in report.points] == [0.25, 0.5, 1.0]
    sizes = [point.n_train for point in report.points]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]
    assert all(0.0 <= point.f1_macro <= 1.0 for point in report.points)
    assert report.model == "logistic_regression"
    assert report.target == "dialect"


def test_curve_is_deterministic(config: ExperimentConfig, tmp_path: Path) -> None:
    first = learning_curve(config, fractions=(0.5, 1.0))
    second = learning_curve(config, fractions=(0.5, 1.0))
    assert first.model_dump() == second.model_dump()

    first.save(tmp_path / "a.json")
    second.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_fractions_are_sorted_and_deduplicated(config: ExperimentConfig) -> None:
    report = learning_curve(config, fractions=(1.0, 0.5, 0.5))
    assert [point.fraction for point in report.points] == [0.5, 1.0]


def test_invalid_fractions_raise(config: ExperimentConfig) -> None:
    with pytest.raises(ConfigurationError, match="at least one fraction"):
        learning_curve(config, fractions=())
    with pytest.raises(ConfigurationError, match=r"\(0, 1\]"):
        learning_curve(config, fractions=(0.0, 0.5))
    with pytest.raises(ConfigurationError, match=r"\(0, 1\]"):
        learning_curve(config, fractions=(0.5, 1.5))


def test_markdown_names_the_curve(config: ExperimentConfig) -> None:
    report = learning_curve(config, fractions=(1.0,))
    markdown = report.to_markdown()
    assert "Learning curve" in markdown
    assert "F1 (macro)" in markdown


def test_cli_learning_curve(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tulip.cli.app import app
    from tulip.config.loader import save_experiment_config

    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="curve-cli")
    config_path = tmp_path / "config.yaml"
    save_experiment_config(config, config_path)

    runner = CliRunner()
    result = runner.invoke(
        app, ["learning-curve", str(config_path), "--fractions", "0.5,1.0", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["points"]) == 2

    saved = runner.invoke(
        app,
        [
            "learning-curve",
            str(config_path),
            "--fractions",
            "1.0",
            "--out",
            str(tmp_path / "curve.json"),
        ],
    )
    assert saved.exit_code == 0, saved.output
    assert (tmp_path / "curve.json").is_file()


def test_cli_rejects_malformed_fractions(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tulip.cli.app import app

    result = CliRunner().invoke(
        app, ["learning-curve", str(tmp_path / "x.yaml"), "--fractions", "abc"]
    )
    assert result.exit_code == 1
    assert "comma-separated numbers" in result.output
