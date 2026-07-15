"""CLI smoke tests for the robustness command."""

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
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="cli-robust")
    path = tmp_path / "experiment.yaml"
    save_experiment_config(config, path)
    return path


def test_robustness_command_runs_and_writes(config_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "robustness"
    result = runner.invoke(
        app,
        [
            "robustness",
            str(config_path),
            "-p",
            "asr_noise",
            "--levels",
            "0,0.5,1.0",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Robustness" in result.output
    assert (out / "robustness-cli-robust.md").is_file()
    assert (out / "robustness-cli-robust.json").is_file()


def test_robustness_rejects_out_of_range_levels(config_path: Path) -> None:
    result = runner.invoke(app, ["robustness", str(config_path), "--levels", "0,2.0"])
    assert result.exit_code == 1
    assert "error:" in result.output
