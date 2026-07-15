"""CLI smoke tests for the openset command and predict --uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from conftest import make_samples
from tulip.cli.app import app
from tulip.config.schemas import SplitConfig
from tulip.data import save_splits
from tulip.data.splitting import speaker_disjoint_split
from tulip.pipeline import DialectClassifier

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

_VOTING = {"name": "voting", "params": {"estimators": ["logistic_regression", "naive_bayes"]}}


def _save_model(model: object, directory: Path) -> Path:
    DialectClassifier(model=model, features=["char_tfidf"], seed=0).fit(
        make_samples(repeats=3)
    ).save(directory)
    return directory


def test_openset_command_runs(tmp_path: Path) -> None:
    splits = speaker_disjoint_split(make_samples(repeats=6), SplitConfig(seed=0))
    model_dir = tmp_path / "model"
    DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=0).fit(
        splits.train
    ).save(model_dir)
    paths = save_splits(splits, tmp_path / "splits")
    result = runner.invoke(
        app, ["openset", str(model_dir), str(paths["validation"]), str(paths["test"])]
    )
    assert result.exit_code == 0, result.output
    assert "Open-set" in result.output


def test_predict_uncertainty_on_ensemble(tmp_path: Path) -> None:
    _save_model(_VOTING, tmp_path / "voting")
    text = make_samples(repeats=1)[0].text
    result = runner.invoke(app, ["predict", str(tmp_path / "voting"), text, "--uncertainty"])
    assert result.exit_code == 0, result.output
    assert "uncertainty" in result.output


def test_predict_uncertainty_rejects_non_ensemble(tmp_path: Path) -> None:
    _save_model("logistic_regression", tmp_path / "logreg")
    result = runner.invoke(
        app, ["predict", str(tmp_path / "logreg"), "jakiś tekst", "--uncertainty"]
    )
    assert result.exit_code == 1
    assert "error:" in result.output
