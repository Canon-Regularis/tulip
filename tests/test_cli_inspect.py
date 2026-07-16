"""CLI smoke tests for the inspect commands (component listings)."""

from __future__ import annotations

from typer.testing import CliRunner

from tulip.cli.app import app

runner = CliRunner()


def test_models_list_names_a_registered_model() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "logistic_regression" in result.output


def test_features_list_names_a_registered_feature() -> None:
    result = runner.invoke(app, ["features", "list"])
    assert result.exit_code == 0, result.output
    assert "char_tfidf" in result.output


def test_explainers_list_names_a_registered_explainer() -> None:
    result = runner.invoke(app, ["explainers", "list"])
    assert result.exit_code == 0, result.output
    # At least one explainer is always registered; the listing must not be empty.
    assert result.output.strip()
