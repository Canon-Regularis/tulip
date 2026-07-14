"""Tests for tulip.evaluation.selective against hand-computed risk-coverage values."""

from __future__ import annotations

import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.predictions import PredictionRecord, SplitPredictions
from tulip.evaluation.selective import risk_coverage_curve, selective_report

# Confidences already descending; correctness T T F T.
#   k=1: risk 0            (top-1 right)
#   k=2: risk 0
#   k=3: risk 1/3          (1 of top-3 wrong)
#   k=4: risk 1/4
# AURC = mean(0, 0, 1/3, 1/4) = 0.145833...
CORRECT = [True, True, False, True]
CONFIDENCE = [0.9, 0.8, 0.7, 0.6]


class TestRiskCoverageCurve:
    def test_hand_computed_curve(self) -> None:
        curve = risk_coverage_curve(CORRECT, CONFIDENCE)
        assert [p.coverage for p in curve] == pytest.approx([0.25, 0.5, 0.75, 1.0])
        assert [p.risk for p in curve] == pytest.approx([0.0, 0.0, 1 / 3, 0.25])
        # Threshold is the lowest confidence still answered at each coverage.
        assert [p.threshold for p in curve] == pytest.approx([0.9, 0.8, 0.7, 0.6])

    def test_confidence_order_is_respected_not_input_order(self) -> None:
        # Same data shuffled: the least-confident wrong sample must land at k=3.
        curve = risk_coverage_curve([False, True, True, True], [0.7, 0.9, 0.6, 0.8])
        assert [p.risk for p in curve] == pytest.approx([0.0, 0.0, 1 / 3, 0.25])

    def test_ties_broken_deterministically(self) -> None:
        # All confidences equal -> order is input order, curve is reproducible.
        a = risk_coverage_curve([True, False, True], [0.5, 0.5, 0.5])
        b = risk_coverage_curve([True, False, True], [0.5, 0.5, 0.5])
        assert [p.risk for p in a] == [p.risk for p in b]

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ConfigurationError, match="same length"):
            risk_coverage_curve([True, False], [0.9])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ConfigurationError, match="zero samples"):
            risk_coverage_curve([], [])


def _predictions(correct: list[bool], confidence: list[float]) -> SplitPredictions:
    """A SplitPredictions whose confidence/correctness match the given arrays."""
    records = []
    for i, (ok, conf) in enumerate(zip(correct, confidence, strict=True)):
        # Two-class distribution with the given top probability; class "a" wins.
        proba = (conf, 1.0 - conf)
        y_true = "a" if ok else "b"
        records.append(PredictionRecord(id=f"s{i}", y_true=y_true, y_pred="a", proba=proba))
    return SplitPredictions(model="m", split="test", labels=("a", "b"), records=tuple(records))


class TestSelectiveReport:
    def test_summary_metrics(self) -> None:
        report = selective_report(_predictions(CORRECT, CONFIDENCE))
        assert report.n_samples == 4
        assert report.aurc == pytest.approx((0.0 + 0.0 + 1 / 3 + 0.25) / 4)
        assert report.base_error == pytest.approx(0.25)

    def test_accuracy_at_coverage(self) -> None:
        report = selective_report(_predictions(CORRECT, CONFIDENCE))
        assert report.accuracy_at_coverage(0.5) == pytest.approx(1.0)  # top-2 both right
        assert report.accuracy_at_coverage(1.0) == pytest.approx(0.75)

    def test_coverage_at_error(self) -> None:
        report = selective_report(_predictions(CORRECT, CONFIDENCE))
        # At <=0 error only the leading all-correct prefix qualifies (coverage 0.5).
        assert report.coverage_at_error(0.0) == pytest.approx(0.5)
        # Allowing 25% error admits the whole set.
        assert report.coverage_at_error(0.25) == pytest.approx(1.0)

    def test_perfect_confidence_beats_base_rate(self) -> None:
        # Confidence perfectly ranks correctness -> low-coverage accuracy is 1.0
        # even though the base rate is only 0.5.
        report = selective_report(_predictions([True, True, False, False], [0.9, 0.8, 0.4, 0.3]))
        assert report.accuracy_at_coverage(0.5) == pytest.approx(1.0)
        assert report.aurc < report.base_error

    def test_markdown_renders(self) -> None:
        report = selective_report(_predictions(CORRECT, CONFIDENCE))
        md = report.to_markdown()
        assert "Selective prediction" in md
        assert "AURC" in md

    def test_rejects_bad_coverage_target(self) -> None:
        with pytest.raises(ConfigurationError, match="coverage target"):
            selective_report(_predictions(CORRECT, CONFIDENCE), coverage_targets=(1.5,))
