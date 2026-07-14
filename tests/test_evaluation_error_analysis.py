"""Tests for tulip.evaluation.error_analysis: confused pairs, exemplars, slices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.evaluation.error_analysis import (
    error_report,
    slice_metrics,
    top_confused_pairs,
)
from tulip.evaluation.predictions import PredictionRecord, SplitPredictions

if TYPE_CHECKING:
    from pathlib import Path

LABELS = ("a", "b", "c")


def _record(
    sample_id: str,
    y_true: str,
    y_pred: str,
    confidence: float,
    *,
    source: str = "syn",
    speaker_id: str = "spk0",
    n_chars: int = 50,
) -> PredictionRecord:
    """A record whose top probability equals ``confidence`` on ``y_pred``."""
    rest = (1.0 - confidence) / (len(LABELS) - 1)
    proba = tuple(confidence if label == y_pred else rest for label in LABELS)
    return PredictionRecord(
        id=sample_id,
        y_true=y_true,
        y_pred=y_pred,
        proba=proba,
        source=source,
        speaker_id=speaker_id,
        n_chars=n_chars,
    )


@pytest.fixture
def predictions() -> SplitPredictions:
    records = (
        _record("s0", "a", "a", 0.9, source="corpusA"),
        _record("s1", "a", "b", 0.8, source="corpusA"),  # wrong, highest conf
        _record("s2", "b", "a", 0.7, source="corpusB"),
        _record("s3", "a", "b", 0.6, source="corpusB"),  # second a->b confusion
        _record("s4", "c", "c", 0.55, source="corpusB"),  # correct, lowest conf
    )
    return SplitPredictions(model="m", split="test", labels=LABELS, records=records)


class TestConfusedPairs:
    def test_ranked_by_count(self, predictions: SplitPredictions) -> None:
        pairs = top_confused_pairs(predictions)
        assert [(p.true_label, p.pred_label, p.count) for p in pairs] == [
            ("a", "b", 2),
            ("b", "a", 1),
        ]

    def test_correct_predictions_are_excluded(self, predictions: SplitPredictions) -> None:
        # No (a, a) or (c, c) on the diagonal appears among confusions.
        pairs = top_confused_pairs(predictions)
        assert all(p.true_label != p.pred_label for p in pairs)


class TestSliceMetrics:
    def test_per_source_accuracy(self, predictions: SplitPredictions) -> None:
        by_source = {s.value: s for s in slice_metrics(predictions) if s.dimension == "source"}
        assert by_source["corpusA"].accuracy == pytest.approx(0.5)  # s0 right, s1 wrong
        assert by_source["corpusB"].accuracy == pytest.approx(1 / 3)  # s4 right of 3
        # Both are below the default low-support threshold.
        assert by_source["corpusA"].low_support and by_source["corpusB"].low_support

    def test_length_band_dimension_present(self, predictions: SplitPredictions) -> None:
        dims = {s.dimension for s in slice_metrics(predictions)}
        assert {"source", "speaker_id", "modality", "length"} <= dims


class TestErrorReport:
    def test_hardest_errors_ordered_by_confidence(self, predictions: SplitPredictions) -> None:
        report = error_report(predictions)
        # Wrong samples s1(0.8), s2(0.7), s3(0.6) -> descending confidence.
        assert [e.id for e in report.hardest_errors] == ["s1", "s2", "s3"]

    def test_least_confident_ordered_ascending(self, predictions: SplitPredictions) -> None:
        report = error_report(predictions)
        assert [e.id for e in report.least_confident][:2] == ["s4", "s3"]

    def test_snippets_added_only_when_texts_supplied(self, predictions: SplitPredictions) -> None:
        plain = error_report(predictions)
        assert all(e.text is None for e in plain.hardest_errors)

        long_text = "x" * 200
        enriched = error_report(predictions, texts={"s1": long_text})
        s1 = next(e for e in enriched.hardest_errors if e.id == "s1")
        assert s1.text is not None and s1.text.endswith("...") and len(s1.text) == 83

    def test_accuracy_matches_overall(self, predictions: SplitPredictions) -> None:
        report = error_report(predictions)
        assert report.accuracy == pytest.approx(2 / 5)  # s0 and s4 correct

    def test_markdown_and_save_round_trip(
        self, predictions: SplitPredictions, tmp_path: Path
    ) -> None:
        report = error_report(predictions)
        assert "Error analysis" in report.to_markdown()
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        report.save(first)
        report.save(second)
        assert first.read_bytes() == second.read_bytes()
