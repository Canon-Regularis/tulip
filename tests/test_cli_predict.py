"""CLI smoke tests for the predict command group."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from conftest import write_manifest_corpus
from tulip.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    from conftest import make_manifest_experiment_config
    from tulip.data import DatasetBuilder
    from tulip.pipeline import DialectClassifier

    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="predict-cli")
    splits = DatasetBuilder(config.data).build(config.split, target=config.target)
    path = tmp_path / "model"
    DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42).fit(
        splits.train
    ).save(path)
    return path


def test_predict_text_json(model_dir: Path) -> None:
    result = runner.invoke(
        app, ["predict", str(model_dir), "hej baca kaj owce pasą na holi", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["label"]
    assert payload["probabilities"]  # a full distribution
    assert all(0.0 <= p["probability"] <= 1.0 for p in payload["probabilities"])


def test_predict_text_table_with_top_k(model_dir: Path) -> None:
    result = runner.invoke(app, ["predict", str(model_dir), "jo żech je z Katowic", "--top-k", "2"])
    assert result.exit_code == 0, result.output
    assert result.output.strip()


def test_predict_uncertainty_needs_an_ensemble(model_dir: Path) -> None:
    # Uncertainty decomposition is an ensemble-only feature; a single logistic
    # model gives a clean error rather than a traceback.
    result = runner.invoke(
        app, ["predict", str(model_dir), "u nos w boru psiwo warzą", "--uncertainty"]
    )
    assert result.exit_code == 1
    assert "ensemble" in result.output


def test_predict_requires_some_input(model_dir: Path) -> None:
    # Neither text nor --audio: a clean error, not a traceback.
    result = runner.invoke(app, ["predict", str(model_dir)])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_explain_global_over_a_corpus(tmp_path: Path) -> None:
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    result = runner.invoke(
        app, ["explain-global", str(corpus / "manifest.csv"), "--top-k", "5", "--json"]
    )
    assert result.exit_code == 0, result.output
    json.loads(result.output)  # a well-formed report


def test_contrast_two_dialects(tmp_path: Path) -> None:
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    result = runner.invoke(
        app,
        [
            "contrast",
            str(corpus / "manifest.csv"),
            "podhale",
            "silesia",
            "--min-support",
            "3",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["dialect_a"] == "podhale" and report["dialect_b"] == "silesia"
    assert report["features"], "expected some distinguishing features"


def test_contrast_rejects_unknown_level(tmp_path: Path) -> None:
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=4, variants=2)
    result = runner.invoke(
        app, ["contrast", str(corpus / "manifest.csv"), "podhale", "silesia", "--level", "bogus"]
    )
    assert result.exit_code != 0
    assert "unknown level" in result.output


def test_fusion_compare_missing_model_errors_cleanly(tmp_path: Path) -> None:
    # A missing model directory must fail with a clean error, not a traceback.
    result = runner.invoke(
        app,
        [
            "fusion-compare",
            str(tmp_path / "no-text"),
            str(tmp_path / "no-audio"),
            str(tmp_path / "data.jsonl"),
        ],
    )
    assert result.exit_code != 0
