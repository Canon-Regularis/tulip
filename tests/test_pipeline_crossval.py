"""Tests for grouped stratified K-fold cross-validation (tulip.pipeline.crossval)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import make_manifest_experiment_config, make_samples, write_manifest_corpus
from tulip.core.exceptions import DataError
from tulip.labels.taxonomy import LabelLevel
from tulip.pipeline import CVConfig, grouped_stratified_kfold, run_cross_validation

if TYPE_CHECKING:
    from pathlib import Path

    from tulip.config import ExperimentConfig


class TestFolds:
    def test_folds_are_speaker_disjoint_and_cover_every_sample(self) -> None:
        samples = make_samples(repeats=6)
        folds = list(grouped_stratified_kfold(samples, k=3, seed=0, target=LabelLevel.DIALECT))
        assert len(folds) == 3
        test_ids: set[str] = set()
        for train, test in folds:
            train_speakers = {s.speaker_id for s in train}
            test_speakers = {s.speaker_id for s in test}
            assert train_speakers.isdisjoint(test_speakers)
            test_ids |= {s.id for s in test}
        # Every labelled (dialect-carrying) sample appears in exactly one test fold.
        labelled = [s for s in samples if s.labels.dialect is not None]
        assert test_ids == {s.id for s in labelled}

    def test_too_few_groups_raises(self) -> None:
        with pytest.raises(DataError, match="distinct"):
            list(
                grouped_stratified_kfold(
                    make_samples(repeats=2), k=20, seed=0, target=LabelLevel.DIALECT
                )
            )


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=3)
    return make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="cv")


class TestRunCrossValidation:
    def test_report_shape_and_bounds(self, config: ExperimentConfig) -> None:
        report = run_cross_validation(config, CVConfig(k=3))
        assert len(report.folds) == 3
        assert {m.metric for m in report.metrics} == {
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
        }
        f1 = report.summary("f1_macro")
        assert 0.0 <= f1.low <= f1.mean <= f1.high <= 1.0

    def test_multi_seed_multiplies_the_runs(self, config: ExperimentConfig) -> None:
        report = run_cross_validation(config, CVConfig(k=3, seeds=(0, 1)))
        assert len(report.folds) == 6
        assert {f.seed for f in report.folds} == {0, 1}

    def test_is_deterministic(self, config: ExperimentConfig) -> None:
        a = run_cross_validation(config, CVConfig(k=3, seeds=(0,)))
        b = run_cross_validation(config, CVConfig(k=3, seeds=(0,)))
        assert a.summary("f1_macro").mean == b.summary("f1_macro").mean

    def test_markdown_renders(self, config: ExperimentConfig) -> None:
        report = run_cross_validation(config, CVConfig(k=3))
        markdown = report.to_markdown()
        assert "Cross-validation" in markdown and "95% CI" in markdown


class TestConfig:
    def test_k_floor(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CVConfig(k=1)

    def test_n_runs(self) -> None:
        assert CVConfig(k=5, seeds=(0, 1, 2)).n_runs == 15
