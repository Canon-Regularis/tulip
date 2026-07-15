"""Tests for `tulip doctor` and the registry discovery commands."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tulip.cli._doctor import (
    _EXTRAS,
    component_statuses,
    probe_extras,
    run_doctor,
)
from tulip.cli.app import app

runner = CliRunner()

# Components that run on the core install; their availability must never depend
# on an optional extra being present.
_CORE_COMPONENTS = ("naive_bayes", "logistic_regression", "char_tfidf", "top_tfidf")


def test_core_components_are_always_available() -> None:
    by_name = {status.name: status for status in component_statuses()}
    for name in _CORE_COMPONENTS:
        assert by_name[name].extra is None
        assert by_name[name].available is True


def test_component_extra_is_read_from_registry_metadata() -> None:
    by_name = {status.name: status for status in component_statuses()}
    assert by_name["herbert"].extra == "transformers"
    assert by_name["wav2vec2_embeddings"].extra == "speech"
    assert by_name["mfcc"].extra == "audio"
    assert by_name["shap"].extra == "explain"
    assert by_name["xgboost"].extra == "boosting"


def test_availability_follows_the_probed_extras() -> None:
    # A component is available iff it needs no extra, or its extra is installed.
    installed = {extra.name for extra in probe_extras() if extra.installed}
    for status in component_statuses():
        expected = status.extra is None or status.extra in installed
        assert status.available is expected, status.name


def test_every_component_extra_is_a_known_extra() -> None:
    # A component must not declare an extra that doctor cannot probe.
    for status in component_statuses():
        if status.extra is not None:
            assert status.extra in _EXTRAS, status.name


def test_report_is_deterministic() -> None:
    assert run_doctor().model_dump() == run_doctor().model_dump()


def test_report_accessors_are_consistent() -> None:
    report = run_doctor()
    # missing_extras is exactly the not-installed subset.
    assert set(report.missing_extras) == {e for e in report.extras if not e.installed}
    # components_of partitions the catalogue by kind.
    models = report.components_of("model")
    assert models and all(c.kind == "model" for c in models)
    assert len(models) < len(report.components)
    # to_markdown mentions the environment and the runnable summary.
    markdown = report.to_markdown()
    assert report.tulip_version in markdown
    assert "runnable now" in markdown


def test_doctor_command_runs() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "components runnable now" in result.output


def test_doctor_json_output_is_valid() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tulip_version"]
    assert {"extras", "components"} <= payload.keys()


def test_models_list_command() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "logistic_regression" in result.output


def test_features_list_shows_both_modalities() -> None:
    result = runner.invoke(app, ["features", "list"])
    assert result.exit_code == 0, result.output
    assert "char_tfidf" in result.output
    assert "mfcc" in result.output


def test_explainers_list_command() -> None:
    result = runner.invoke(app, ["explainers", "list"])
    assert result.exit_code == 0, result.output
    assert "top_tfidf" in result.output
