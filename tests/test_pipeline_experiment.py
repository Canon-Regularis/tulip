"""End-to-end tests for the experiment and benchmark runners + example configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_manifest_experiment_config, write_manifest_corpus
from tulip.config import ExperimentConfig, load_experiment_config
from tulip.evaluation.benchmark import load_benchmark
from tulip.pipeline import run_benchmark, run_experiment

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def experiment_config(tmp_path: Path) -> ExperimentConfig:
    """A tiny but complete text experiment over an on-disk manifest corpus."""
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=5, variants=3)
    return make_manifest_experiment_config(corpus, tmp_path / "artifacts")


class TestBuildClassifierTrainingKnobs:
    """build_classifier injects only the TrainingConfig knobs a model declares.

    The model's metadata is stubbed rather than a probe model registered, so the
    global model registry (and other tests asserting over it) is untouched. A
    real raw-input model name (``fasttext``) keeps the config valid; only the
    metadata lookup inside build_classifier is redirected.
    """

    def _config(self, tmp_path: Path) -> ExperimentConfig:
        from tulip.config.schemas import ComponentConfig, TrainingConfig

        corpus = write_manifest_corpus(tmp_path / "corpus", speakers=5, variants=2)
        return make_manifest_experiment_config(
            corpus,
            tmp_path / "artifacts",
            model=ComponentConfig(name="fasttext"),
            features=[],
            training=TrainingConfig(batch_size=9, epochs=7, learning_rate=0.3),
        )

    def test_default_injects_all_three_knobs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tulip.models import MODELS
        from tulip.pipeline.experiment import build_classifier

        config = self._config(tmp_path)
        monkeypatch.setattr(MODELS, "metadata", lambda name: {"training_aware": True})
        classifier = build_classifier(config)
        assert classifier.model_config.params == {
            "batch_size": 9,
            "epochs": 7,
            "learning_rate": 0.3,
        }

    def test_declared_knobs_restrict_the_injection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The OCP fix: a training-aware model that accepts only ``epochs`` gets
        # only that knob, not the neural fine-tuners' full set.
        from tulip.models import MODELS
        from tulip.pipeline.experiment import build_classifier

        config = self._config(tmp_path)
        monkeypatch.setattr(
            MODELS,
            "metadata",
            lambda name: {"training_aware": True, "training_knobs": ("epochs",)},
        )
        classifier = build_classifier(config)
        assert classifier.model_config.params == {"epochs": 7}


class TestRunExperiment:
    def test_end_to_end_artifacts_and_reports(self, experiment_config: ExperimentConfig) -> None:
        result = run_experiment(experiment_config)

        assert result.name == "mini-text"
        assert result.sizes["train"] > 0 and result.sizes["test"] > 0
        assert "test" in result.reports
        assert 0.0 <= result.reports["test"].accuracy <= 1.0
        assert result.reports["test"].n_samples == result.sizes["test"]
        assert "mini-text" in result.summary()

        out = experiment_config.output_dir / "mini-text"
        for artifact in (
            "model/model.joblib",
            "model/metadata.json",
            "splits/train.jsonl",
            "splits/build_manifest.json",
            "config.yaml",
            "report_test.json",
            "result.json",
        ):
            assert (out / artifact).is_file(), artifact

        # The persisted config must round-trip through the loader.
        reloaded = load_experiment_config(out / "config.yaml")
        assert reloaded.name == experiment_config.name
        assert reloaded.model.name == "logistic_regression"

        summary = json.loads((out / "result.json").read_text(encoding="utf-8"))
        assert summary["model"] == "logistic_regression"
        assert summary["reports"]["test"]["accuracy"] == result.reports["test"].accuracy

    def test_saved_model_is_loadable_and_predicts(
        self, experiment_config: ExperimentConfig
    ) -> None:
        from tulip.pipeline import DialectClassifier

        result = run_experiment(experiment_config)
        classifier = DialectClassifier.load(result.model_path)
        prediction = classifier.predict("Hej baca się pyto kaj się owce pasą na holi.")
        assert prediction.label == "podhale"


class TestRunBenchmark:
    def test_models_compared_on_identical_frozen_split(
        self, experiment_config: ExperimentConfig
    ) -> None:
        results = run_benchmark(experiment_config, models=["naive_bayes", "logistic_regression"])

        assert [r.model for r in results] == ["naive_bayes", "logistic_regression"]
        assert len({r.n_train for r in results}) == 1  # identical split for all
        assert len({r.n_test for r in results}) == 1
        for result in results:
            assert "test" in result.reports
            assert result.wall_seconds >= 0

        out = experiment_config.output_dir / "mini-text"
        assert (out / "benchmark.json").is_file()
        markdown = (out / "benchmark.md").read_text(encoding="utf-8")
        assert "naive_bayes" in markdown and "logistic_regression" in markdown

        reloaded = load_benchmark(out / "benchmark.json")
        assert {r.model for r in reloaded} == {"naive_bayes", "logistic_regression"}

    def test_parallel_benchmark_matches_sequential(
        self, experiment_config: ExperimentConfig
    ) -> None:
        # Each model re-seeds its own process, so process-parallel training yields
        # the same models in the same order with the same metrics (only the
        # machine-dependent wall_seconds may differ, and it never enters the board).
        models = ["naive_bayes", "logistic_regression", "linear_svm"]
        sequential = run_benchmark(experiment_config, models=models, n_jobs=1)
        parallel = run_benchmark(experiment_config, models=models, n_jobs=2)

        def board_view(results: list) -> list:
            return [(r.model, r.n_train, r.n_test, r.reports["test"].model_dump()) for r in results]

        assert board_view(sequential) == board_view(parallel)

    @pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.yaml")), ids=lambda p: p.name)
    def test_every_shipped_config_validates(self, path: Path) -> None:
        config = load_experiment_config(path)
        assert config.name
        assert config.model.name

    def test_configs_directory_is_not_empty(self) -> None:
        assert list(CONFIGS_DIR.glob("*.yaml")), "example configs must ship with the repo"
