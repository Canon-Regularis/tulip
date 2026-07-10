"""Tests for the tulip command-line interface."""

from __future__ import annotations

import json
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
def mini_config(tmp_path: Path) -> Path:
    """A complete experiment config YAML over a temp manifest corpus.

    Written through save_experiment_config, so the CLI tests also exercise
    the config save -> load round trip end to end.
    """
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=5, variants=2)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="cli-mini")
    config_path = tmp_path / "experiment.yaml"
    save_experiment_config(config, config_path)
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

    def test_download_reports_auto_and_manual(self, fake_bigos_hub, tmp_path: Path) -> None:
        result = runner.invoke(app, ["data", "download", "bigos", "dgp", "--root", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "downloaded" in result.output
        assert "manual" in result.output
        assert (tmp_path / "bigos" / "manifest.csv").is_file()

    def test_download_requires_names_or_all(self) -> None:
        result = runner.invoke(app, ["data", "download"])
        assert result.exit_code == 1
        assert "name at least one corpus" in result.output

    def test_download_failure_exits_nonzero_but_still_renders_table(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import sys
        from types import ModuleType

        def gated(*args, **kwargs):
            raise RuntimeError("gated dataset on the Hub. You must be authenticated")

        module = ModuleType("datasets")
        module.load_dataset = gated
        monkeypatch.setitem(sys.modules, "datasets", module)

        result = runner.invoke(app, ["data", "download", "bigos", "dgp", "--root", str(tmp_path)])
        assert result.exit_code == 1
        assert "failed" in result.output
        assert "manual" in result.output  # dgp's guidance still rendered

    def test_prepare_builds_splits(self, mini_config: Path, tmp_path: Path) -> None:
        out = tmp_path / "splits"
        result = runner.invoke(app, ["data", "prepare", str(mini_config), "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "train.jsonl").is_file()
        assert "train:" in result.output


class TestSynthesize:
    def test_synthesize_writes_a_labelled_manifest(self, tmp_path: Path) -> None:
        out = tmp_path / "synth"
        result = runner.invoke(
            app,
            ["data", "synthesize", "--out", str(out), "--speakers", "2", "--per-speaker", "2"],
        )
        assert result.exit_code == 0, result.output
        assert (out / "manifest.jsonl").is_file()
        # The per-class table must show real labels, not "__unlabelled__": that
        # would mean the written manifest is being read back as a split file.
        assert "__unlabelled__" not in result.output
        assert "podhale" in result.output

    def test_synthesize_is_reproducible_for_a_seed(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        for out in (first, second):
            result = runner.invoke(
                app,
                ["data", "synthesize", "--out", str(out), "--seed", "7", "--speakers", "2"],
            )
            assert result.exit_code == 0, result.output
        left = (first / "manifest.jsonl").read_bytes()
        assert left == (second / "manifest.jsonl").read_bytes()

    def test_synthesize_audio_writes_clips_and_a_manifest(self, tmp_path: Path) -> None:
        out = tmp_path / "audio"
        result = runner.invoke(
            app,
            [
                "data",
                "synthesize-audio",
                "--out",
                str(out),
                "--speakers",
                "2",
                "--per-speaker",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "manifest.jsonl").is_file()
        assert list((out / "audio").glob("*.wav")), "no WAV clips were written"
        assert "__unlabelled__" not in result.output
        assert "podhale" in result.output


class TestValidate:
    def test_a_clean_manifest_exits_zero(self, tmp_path: Path) -> None:
        out = tmp_path / "synth"
        runner.invoke(app, ["data", "synthesize", "--out", str(out), "--speakers", "2"])
        result = runner.invoke(app, ["data", "validate", str(out / "manifest.jsonl")])
        assert result.exit_code == 0, result.output

    def test_a_broken_manifest_exits_one(self, tmp_path: Path) -> None:
        """The CI contract: `data validate` must gate on errors."""
        manifest = tmp_path / "bad.csv"
        manifest.write_text("id,speaker_id\n1,spk1\n", encoding="utf-8")
        result = runner.invoke(app, ["data", "validate", str(manifest)])
        assert result.exit_code == 1
        assert "text" in result.output  # names the missing column

    def test_json_output_is_parseable(self, tmp_path: Path) -> None:
        out = tmp_path / "synth"
        runner.invoke(app, ["data", "synthesize", "--out", str(out), "--speakers", "2"])
        result = runner.invoke(app, ["data", "validate", str(out / "manifest.jsonl"), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["n_rows"] > 0


class TestCards:
    def test_model_card_renders_from_a_trained_artifact(
        self, mini_config: Path, tmp_path: Path
    ) -> None:
        assert runner.invoke(app, ["train", str(mini_config)]).exit_code == 0
        model_dir = tmp_path / "artifacts" / "cli-mini" / "model"

        result = runner.invoke(app, ["card", "model", str(model_dir)])
        assert result.exit_code == 0, result.output
        assert "Model card" in result.output

    def test_card_commands_never_leak_a_traceback(self, tmp_path: Path) -> None:
        """File readers raise FileNotFoundError/ValidationError, which are not TulipErrors.

        They escaped `_tulip_errors` as a rich traceback that leaked absolute
        filesystem paths, unlike every other file-reading command.
        """
        (tmp_path / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
        (tmp_path / "bad_report.json").write_text('{"not": "a report"}', encoding="utf-8")

        invocations = [
            ["card", "dataset", str(tmp_path / "missing.json")],
            ["card", "dataset", str(tmp_path / "list.json")],
            ["card", "model", str(tmp_path / "no_such_model_dir")],
        ]
        for argv in invocations:
            result = runner.invoke(app, argv)
            assert result.exit_code == 1, argv
            assert "Traceback" not in result.output, argv
            assert "error:" in result.output, argv

    def test_card_model_rejects_a_malformed_report(self, mini_config: Path, tmp_path: Path) -> None:
        assert runner.invoke(app, ["train", str(mini_config)]).exit_code == 0
        model_dir = tmp_path / "artifacts" / "cli-mini" / "model"
        bad = tmp_path / "bad_report.json"
        bad.write_text('{"not": "a report"}', encoding="utf-8")

        result = runner.invoke(app, ["card", "model", str(model_dir), "--report", str(bad)])
        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "not a valid evaluation report" in result.output

    def test_dataset_card_writes_to_a_file(self, mini_config: Path, tmp_path: Path) -> None:
        assert runner.invoke(app, ["train", str(mini_config)]).exit_code == 0
        manifest = tmp_path / "artifacts" / "cli-mini" / "splits" / "build_manifest.json"
        destination = tmp_path / "DATASET_CARD.md"

        result = runner.invoke(
            app,
            ["card", "dataset", str(manifest), "--dataset", "manifest", "--out", str(destination)],
        )
        assert result.exit_code == 0, result.output
        assert destination.read_text(encoding="utf-8").startswith("# Dataset card")


class TestSelfTrain:
    def test_selftrain_reports_pseudo_label_rounds(self, mini_config: Path, tmp_path: Path) -> None:
        out = tmp_path / "splits"
        assert (
            runner.invoke(
                app, ["data", "prepare", str(mini_config), "--output", str(out)]
            ).exit_code
            == 0
        )

        result = runner.invoke(
            app,
            [
                "selftrain",
                str(out / "train.jsonl"),
                str(out / "test.jsonl"),
                "-f",
                "char_tfidf",
                "--threshold",
                "0.5",
                "--iters",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "pseudo-label" in result.output

    def test_missing_features_fails_with_guidance_not_a_traceback(
        self, mini_config: Path, tmp_path: Path
    ) -> None:
        """A classical model handed raw text dies deep in sklearn; catch it early."""
        out = tmp_path / "splits"
        runner.invoke(app, ["data", "prepare", str(mini_config), "--output", str(out)])

        result = runner.invoke(
            app, ["selftrain", str(out / "train.jsonl"), str(out / "test.jsonl")]
        )
        assert result.exit_code == 1
        assert "char_tfidf" in result.output  # tells the user what to pass
        assert "--raw" in result.output

    def test_raw_with_a_classical_model_fails_cleanly(
        self, mini_config: Path, tmp_path: Path
    ) -> None:
        """`--raw` defaults to logistic_regression, which cannot take raw text.

        This used to sail past the guard and die inside sklearn with
        `could not convert string to float`, escaping the TulipError boundary as
        a full traceback.
        """
        out = tmp_path / "splits"
        runner.invoke(app, ["data", "prepare", str(mini_config), "--output", str(out)])

        result = runner.invoke(
            app, ["selftrain", str(out / "train.jsonl"), str(out / "test.jsonl"), "--raw"]
        )
        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "could not convert string to float" not in result.output
        assert "cannot consume raw text input" in result.output

    def test_raw_and_feature_are_mutually_exclusive(
        self, mini_config: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "splits"
        runner.invoke(app, ["data", "prepare", str(mini_config), "--output", str(out)])

        result = runner.invoke(
            app,
            [
                "selftrain",
                str(out / "train.jsonl"),
                str(out / "test.jsonl"),
                "--raw",
                "-f",
                "char_tfidf",
            ],
        )
        assert result.exit_code == 1
        assert "drop one of them" in result.output


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

    def test_predict_json_output_is_parseable(self, trained_text_artifact: Path) -> None:
        result = runner.invoke(
            app, ["predict", str(trained_text_artifact), "Godom po naszymu cołki czos.", "--json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["label"] == "silesia"
        assert len(payload["probabilities"]) == 3

    def test_predict_with_explanation(self, trained_text_artifact: Path) -> None:
        result = runner.invoke(
            app,
            [
                "predict",
                str(trained_text_artifact),
                "Kaj żeś boł wczorej?",
                "--explain",
                "top_tfidf",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "evidence" in result.output

    def test_predict_requires_exactly_one_input(self, trained_text_artifact: Path) -> None:
        result = runner.invoke(app, ["predict", str(trained_text_artifact)])
        assert result.exit_code == 1
        assert "exactly one input" in result.output

    def test_standalone_explain_command(self, trained_text_artifact: Path) -> None:
        result = runner.invoke(
            app,
            [
                "explain",
                str(trained_text_artifact),
                "Kaj żeś boł wczorej?",
                "--method",
                "top_tfidf",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "evidence" in result.output

    def test_explain_requires_exactly_one_input(self, trained_text_artifact: Path) -> None:
        result = runner.invoke(app, ["explain", str(trained_text_artifact)])
        assert result.exit_code == 1
        assert "exactly one input" in result.output

    def test_audio_input_on_a_text_model_fails_cleanly(
        self, trained_text_artifact: Path, tmp_path: Path
    ) -> None:
        """Feeding audio to a text model used to die deep in the feature stack.

        The mismatch escaped the TulipError boundary as a raw traceback; it now
        surfaces as one clean error line, for both `predict` and `explain`.
        """
        clip = tmp_path / "x.wav"
        clip.write_bytes(b"RIFF....")
        for command in ("predict", "explain"):
            result = runner.invoke(app, [command, str(trained_text_artifact), "--audio", str(clip)])
            assert result.exit_code == 1, command
            assert "Traceback" not in result.output, command
            assert "not audio" in result.output, command

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
