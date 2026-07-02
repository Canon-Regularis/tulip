"""End-to-end tests for the experiment and benchmark runners + example configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tulip.config import ExperimentConfig, load_experiment_config
from tulip.config.schemas import ComponentConfig, DataConfig, SplitConfig
from tulip.evaluation.benchmark import load_benchmark
from tulip.pipeline import run_benchmark, run_experiment

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def experiment_config(tmp_path: Path) -> ExperimentConfig:
    """A tiny but complete text experiment over an on-disk manifest corpus."""
    rows = ["id,text,speaker_id,dialect"]
    texts = {
        "podhale": "Hej baca się pyto kaj się owce pasą na holi wariant {i}.",
        "silesia": "Jo żech je z Katowic i godom po naszymu cołki czos wariant {i}.",
        "kurpie": "U nos w boru psiwo warzą jesce po staremu wariant {i}.",
    }
    for dialect, template in texts.items():
        for speaker in range(5):
            for i in range(3):
                text = template.format(i=f"{speaker}-{i}")
                rows.append(f"{dialect}-{speaker}-{i},{text},{dialect}-spk{speaker},{dialect}")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    return ExperimentConfig(
        name="mini-text",
        seed=42,
        data=DataConfig(
            datasets=[ComponentConfig(name="manifest", params={"root": str(corpus)})],
            root=tmp_path,
            deduplicate=False,  # "wariant N" texts are intentionally similar
            min_text_chars=10,
        ),
        features=[ComponentConfig(name="char_tfidf")],
        model=ComponentConfig(name="logistic_regression"),
        split=SplitConfig(seed=42),
        output_dir=tmp_path / "artifacts",
    )


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


class TestExampleConfigs:
    @pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.yaml")), ids=lambda p: p.name)
    def test_every_shipped_config_validates(self, path: Path) -> None:
        config = load_experiment_config(path)
        assert config.name
        assert config.model.name

    def test_configs_directory_is_not_empty(self) -> None:
        assert list(CONFIGS_DIR.glob("*.yaml")), "example configs must ship with the repo"
