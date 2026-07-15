"""Tests for subgroup fairness disparity."""

from __future__ import annotations

import pytest

from tulip.evaluation.fairness import fairness_report, worst_group_gap
from tulip.evaluation.predictions import PredictionRecord, SplitPredictions

_LABELS = ("a", "b", "c")
_CYCLE = ["a", "b", "c"]


def _record(
    sample_id: str, y_true: str, y_pred: str, source: str, speaker: str
) -> PredictionRecord:
    confidence = 0.8
    rest = (1.0 - confidence) / 2
    proba = tuple(confidence if label == y_pred else rest for label in _LABELS)
    return PredictionRecord(
        id=sample_id,
        y_true=y_true,
        y_pred=y_pred,
        proba=proba,
        source=source,
        speaker_id=speaker,
        n_chars=40,
    )


def _group(prefix: str, source: str, size: int, *, correct: bool) -> list[PredictionRecord]:
    records = []
    for i in range(size):
        truth = _CYCLE[i % 3]
        pred = truth if correct else ("b" if truth == "a" else "a")
        records.append(_record(f"{prefix}{i}", truth, pred, source, f"{source}-spk{i % 2}"))
    return records


def _predictions(records: list[PredictionRecord]) -> SplitPredictions:
    return SplitPredictions(model="m", split="test", labels=_LABELS, records=tuple(records))


def test_detects_strong_source_disparity() -> None:
    predictions = _predictions(
        _group("A", "corpusA", 12, correct=True) + _group("B", "corpusB", 12, correct=False)
    )
    report = fairness_report(predictions)
    source = next(d for d in report.dimensions if d.dimension == "source")
    assert source.worst_group == "corpusB"
    assert source.best_group == "corpusA"
    assert source.gap == pytest.approx(1.0)
    assert source.ratio == pytest.approx(0.0)
    assert source.significant
    assert not source.worst_low_support


def test_balanced_data_has_no_gap() -> None:
    predictions = _predictions(
        _group("A", "corpusA", 12, correct=True) + _group("B", "corpusB", 12, correct=True)
    )
    source = next(d for d in fairness_report(predictions).dimensions if d.dimension == "source")
    assert source.gap == pytest.approx(0.0)
    assert not source.significant


def test_low_support_worst_group_is_flagged() -> None:
    predictions = _predictions(
        _group("C", "common", 12, correct=True) + _group("R", "rare", 3, correct=False)
    )
    source = next(d for d in fairness_report(predictions).dimensions if d.dimension == "source")
    assert source.worst_group == "rare"
    assert source.worst_low_support  # 3 < DEFAULT_LOW_SUPPORT


def test_worst_group_gap_prefers_a_reliable_dimension() -> None:
    # source is a reliable gap of 0.5 (both corpora n=6). The speaker dimension
    # has a wider gap of 1.0 but its worst speaker is low-support, so the headline
    # stays on the reliable source.
    records = _group("A", "corpusA", 6, correct=True)
    for i in range(3):  # corpusB-spk0: all correct
        records.append(_record(f"B0{i}", _CYCLE[i], _CYCLE[i], "corpusB", "corpusB-spk0"))
    for i in range(3):  # corpusB-spk1: all wrong
        truth = _CYCLE[i]
        wrong = "b" if truth == "a" else "a"
        records.append(_record(f"B1{i}", truth, wrong, "corpusB", "corpusB-spk1"))
    report = fairness_report(_predictions(records))

    source = next(d for d in report.dimensions if d.dimension == "source")
    speaker = next(d for d in report.dimensions if d.dimension == "speaker_id")
    assert source.gap == pytest.approx(0.5)
    assert not source.worst_low_support
    assert speaker.gap == pytest.approx(1.0)
    assert speaker.worst_low_support

    headline = worst_group_gap(report)
    assert headline is not None
    assert headline.dimension == "source"  # the reliable one, not the wider low-support gap
    assert report.max_gap == pytest.approx(0.5)


def test_report_is_byte_stable() -> None:
    predictions = _predictions(
        _group("A", "corpusA", 12, correct=True) + _group("B", "corpusB", 12, correct=False)
    )
    assert fairness_report(predictions).model_dump() == fairness_report(predictions).model_dump()


def test_markdown_names_the_report() -> None:
    predictions = _predictions(_group("A", "corpusA", 6, correct=True))
    report = fairness_report(predictions)
    assert "Fairness" in report.to_markdown()
