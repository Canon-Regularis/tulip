"""Tests for the tulip command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import make_samples
from tulip.cli.app import app
from tulip.pipeline import DialectClassifier

runner = CliRunner()


@pytest.fixture(scope="module")
def model_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small trained text model saved to disk once per module."""
    artifact = tmp_path_factory.mktemp("cli-model") / "model"
    classifier = DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42)
    classifier.fit(make_samples())
    classifier.save(artifact)
    return artifact


@pytest.fixture
def mini_config(tmp_path: Path) -> Path:
    """A complete experiment config over a temp manifest corpus."""
    rows = ["id,text,speaker_id,dialect"]
    texts = {
        "podhale": "Hej baca się pyto kaj się owce pasą na holi wariant {i}.",
        "silesia": "Jo żech je z Katowic i godom po naszymu cołki czos wariant {i}.",
        "kurpie": "U nos w boru psiwo warzą jesce po staremu wariant {i}.",
    }
    for dialect, template in texts.items():
        for speaker in range(5):
            for i in range(2):
                text = template.format(i=f"{speaker}-{i}")
                rows.append(f"{dialect}-{speaker}-{i},{text},{dialect}-spk{speaker},{dialect}")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    config = f"""
name: cli-mini
seed: 42
task: text
target: dialect
data:
  root: {json.dumps(str(tmp_path))}
  datasets:
    - name: manifest
      params:
        root: {json.dumps(str(corpus))}
  deduplicate: false
  min_text_chars: 10
features:
  - name: char_tfidf
model:
  name: logistic_regression
split:
  seed: 42
output_dir: {json.dumps(str(tmp_path / "artifacts"))}
"""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(config, encoding="utf-8")
    return config_path


class TestBasics:
    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "predict" in result.output

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "tulip" in result.output


class TestDatasets:
    def test_list_shows_catalog_with_tiers(self) -> None:
        result = runner.invoke(app, ["data", "list"])
        assert result.exit_code == 0
        assert "dialektarium" in result.output
        assert "docs/datasets.md" in result.output

    def test_prepare_builds_splits(self, mini_config: Path, tmp_path: Path) -> None:
        out = tmp_path / "splits"
        result = runner.invoke(app, ["data", "prepare", str(mini_config), "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "train.jsonl").is_file()
        assert "train:" in result.output


class TestTrainAndPredict:
    def test_train_then_predict_round_trip(self, mini_config: Path, tmp_path: Path) -> None:
        result = runner.invoke(app, ["train", str(mini_config)])
        assert result.exit_code == 0, result.output
        assert "cli-mini" in result.output

        model_dir = tmp_path / "artifacts" / "cli-mini" / "model"
        predicted = runner.invoke(
            app, ["predict", str(model_dir), "Hej baca się pyto kaj się owce pasą."]
        )
        assert predicted.exit_code == 0, predicted.output
        assert "podhale" in predicted.output

    def test_predict_json_output_is_parseable(self, model_artifact: Path) -> None:
        result = runner.invoke(
            app, ["predict", str(model_artifact), "Godom po naszymu cołki czos.", "--json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["label"] == "silesia"
        assert len(payload["probabilities"]) == 3

    def test_predict_with_explanation(self, model_artifact: Path) -> None:
        result = runner.invoke(
            app,
            ["predict", str(model_artifact), "Kaj żeś boł wczorej?", "--explain", "top_tfidf"],
        )
        assert result.exit_code == 0, result.output
        assert "evidence" in result.output

    def test_predict_requires_exactly_one_input(self, model_artifact: Path) -> None:
        result = runner.invoke(app, ["predict", str(model_artifact)])
        assert result.exit_code == 1
        assert "exactly one input" in result.output

    def test_missing_model_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["predict", str(tmp_path / "nope"), "tekst"])
        assert result.exit_code == 1
        assert "error:" in result.output


class TestEvaluateAndBenchmark:
    def test_evaluate_on_split_file(self, mini_config: Path, tmp_path: Path) -> None:
        assert runner.invoke(app, ["train", str(mini_config)]).exit_code == 0
        model_dir = tmp_path / "artifacts" / "cli-mini" / "model"
        test_split = tmp_path / "artifacts" / "cli-mini" / "splits" / "test.jsonl"
        result = runner.invoke(app, ["evaluate", str(model_dir), str(test_split), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert 0.0 <= payload["accuracy"] <= 1.0

    def test_benchmark_compares_models(self, mini_config: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["benchmark", str(mini_config), "-m", "naive_bayes", "-m", "logistic_regression"],
        )
        assert result.exit_code == 0, result.output
        assert "benchmark" in result.output
        # The rendered rich table truncates cells at the runner's 80-char
        # terminal; the persisted markdown carries the authoritative names.
        markdown = (tmp_path / "artifacts" / "cli-mini" / "benchmark.md").read_text("utf-8")
        assert "naive_bayes" in markdown
        assert "logistic_regression" in markdown
