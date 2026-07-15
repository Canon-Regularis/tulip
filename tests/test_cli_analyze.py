"""CLI tests for analyze --hierarchical and --power."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from tulip.cli.app import app
from tulip.evaluation.predictions import PredictionRecord, SplitPredictions

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

_LABELS = ("kashubia", "podhale", "spisz")


def _record(sample_id: str, y_true: str, y_pred: str, confidence: float = 0.8) -> PredictionRecord:
    rest = (1.0 - confidence) / (len(_LABELS) - 1)
    proba = tuple(confidence if label == y_pred else rest for label in _LABELS)
    return PredictionRecord(
        id=sample_id,
        y_true=y_true,
        y_pred=y_pred,
        proba=proba,
        source="syn",
        speaker_id="spk0",
        n_chars=40,
    )


def _save_predictions(path: Path) -> None:
    records = (
        _record("s0", "podhale", "podhale"),
        _record("s1", "podhale", "spisz"),  # family match, wrong dialect
        _record("s2", "spisz", "podhale"),
        _record("s3", "kashubia", "kashubia"),
        _record("s4", "podhale", "kashubia"),  # family mismatch
    )
    SplitPredictions(model="m", split="test", labels=_LABELS, records=records).save(path)


def test_analyze_hierarchical_and_power_markdown(tmp_path: Path) -> None:
    path = tmp_path / "predictions_test.json"
    _save_predictions(path)
    result = runner.invoke(app, ["analyze", str(path), "--hierarchical", "--power"])
    assert result.exit_code == 0, result.output
    assert "Hierarchical metrics" in result.output
    assert "power" in result.output.lower()


def test_analyze_hierarchical_and_power_json(tmp_path: Path) -> None:
    path = tmp_path / "predictions_test.json"
    _save_predictions(path)
    result = runner.invoke(app, ["analyze", str(path), "--hierarchical", "--power", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "hierarchical" in payload
    assert "power" in payload


def test_analyze_fairness(tmp_path: Path) -> None:
    path = tmp_path / "predictions_test.json"
    _save_predictions(path)
    result = runner.invoke(app, ["analyze", str(path), "--fairness"])
    assert result.exit_code == 0, result.output
    assert "Fairness" in result.output

    result_json = runner.invoke(app, ["analyze", str(path), "--fairness", "--json"])
    assert result_json.exit_code == 0, result_json.output
    assert "fairness" in json.loads(result_json.output)
