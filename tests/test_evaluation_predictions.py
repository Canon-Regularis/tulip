"""Tests for the per-sample prediction substrate (SplitPredictions + collector)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from conftest import make_samples
from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.predictions import PredictionRecord, SplitPredictions
from tulip.pipeline import DialectClassifier
from tulip.pipeline.experiment import collect_predictions, evaluate_samples

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fitted_classifier() -> DialectClassifier:
    classifier = DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42)
    classifier.fit(make_samples())
    return classifier


class TestCollectPredictions:
    def test_records_align_with_the_aggregate_report(
        self, fitted_classifier: DialectClassifier
    ) -> None:
        samples = make_samples()
        report = evaluate_samples(fitted_classifier, samples, name="test")
        predictions = collect_predictions(fitted_classifier, samples, name="test")

        # One record per evaluated (non-skipped) sample, and correctness agrees
        # with the report's accuracy exactly (same raw-argmax scoring).
        assert len(predictions) == report.n_samples
        assert predictions.correct().mean() == pytest.approx(report.accuracy)
        assert predictions.labels == fitted_classifier.classes_
        assert predictions.model == "logistic_regression"
        assert predictions.split == "test"

    def test_probabilities_and_argmax_are_consistent(
        self, fitted_classifier: DialectClassifier
    ) -> None:
        predictions = collect_predictions(fitted_classifier, make_samples(), name="test")
        matrix = predictions.proba_matrix()
        assert matrix.shape == (len(predictions), len(predictions.labels))
        # Each stored prediction is the argmax over its own probability row.
        argmax_labels = [predictions.labels[i] for i in np.argmax(matrix, axis=1)]
        assert argmax_labels == predictions.pred_labels()
        # confidence == row max.
        np.testing.assert_allclose(predictions.confidences(), matrix.max(axis=1), atol=1e-9)

    def test_records_carry_self_describing_slice_keys(
        self, fitted_classifier: DialectClassifier
    ) -> None:
        predictions = collect_predictions(fitted_classifier, make_samples(), name="test")
        record = predictions.records[0]
        assert record.source == "synthetic"
        assert record.speaker_id is not None
        assert record.modality == "text"
        assert record.n_chars is not None and record.n_chars > 0


class TestPersistenceRoundTrip:
    def test_save_load_round_trips(
        self, fitted_classifier: DialectClassifier, tmp_path: Path
    ) -> None:
        predictions = collect_predictions(fitted_classifier, make_samples(), name="test")
        path = tmp_path / "predictions_test.json"
        predictions.save(path)
        reloaded = SplitPredictions.load(path)
        assert reloaded.model == predictions.model
        assert reloaded.labels == predictions.labels
        assert reloaded.pred_labels() == predictions.pred_labels()
        assert reloaded.true_labels() == predictions.true_labels()

    def test_dump_is_byte_identical_across_writes(
        self, fitted_classifier: DialectClassifier, tmp_path: Path
    ) -> None:
        predictions = collect_predictions(fitted_classifier, make_samples(), name="test")
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        predictions.save(first)
        predictions.save(second)
        assert first.read_bytes() == second.read_bytes()

    def test_load_rejects_a_non_predictions_file(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.json"
        bogus.write_text('{"not": "predictions"}', encoding="utf-8")
        with pytest.raises(ConfigurationError):
            SplitPredictions.load(bogus)


class TestSplitPredictionsValidation:
    def test_rejects_misaligned_probability_row(self) -> None:
        with pytest.raises(ValueError, match="align with labels"):
            SplitPredictions(
                model="m",
                split="test",
                labels=("a", "b", "c"),
                records=(PredictionRecord(id="s1", y_true="a", y_pred="a", proba=(0.5, 0.5)),),
            )

    def test_rejects_duplicate_labels(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            SplitPredictions(model="m", split="test", labels=("a", "a"), records=())

    def test_record_confidence_and_correctness(self) -> None:
        record = PredictionRecord(id="s1", y_true="a", y_pred="b", proba=(0.3, 0.7))
        assert record.confidence == pytest.approx(0.7)
        assert record.correct is False
