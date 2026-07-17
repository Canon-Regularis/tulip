"""Tests for geographic and demographic slice keys and their downstream slicing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.core.types import DialectLabels, Sample
from tulip.evaluation.error_analysis import slice_metrics
from tulip.evaluation.fairness import fairness_report
from tulip.evaluation.predictions import PredictionRecord, SplitPredictions
from tulip.evaluation.slicing import age_band, record_slice_keys

if TYPE_CHECKING:
    from pathlib import Path

_LABELS = ("podhale", "spisz")


class TestAgeBand:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (10, "<=19"),
            (19, "<=19"),
            (20, "20-29"),
            (34, "30-44"),
            (44, "30-44"),
            (45, "45-59"),
            (60, "60+"),
            (90, "60+"),
            ("34", "30-44"),
            ("34.0", "30-44"),
        ],
    )
    def test_numeric_ages_bucket(self, value: object, expected: str) -> None:
        assert age_band(value) == expected

    def test_non_numeric_passes_through_stripped(self) -> None:
        # Common Voice ships decade strings, not integers; keep the corpus's bands.
        assert age_band(" thirties ") == "thirties"
        assert age_band("50-59") == "50-59"


class TestRecordSliceKeys:
    def test_geographic_from_labels_demographic_from_metadata(self) -> None:
        sample = Sample(
            id="s1",
            text="hej",
            speaker_id="spk",
            source="dialektarium",
            labels=DialectLabels(dialect="podhale"),
            metadata={"age": "34", "gender": "F"},
        )
        keys = record_slice_keys(sample)
        assert keys == {
            "family": "lesser_polish",  # auto-derived from dialect
            "dialect": "podhale",
            "region": None,
            "voivodeship": None,
            "gender": "f",  # lowercased
            "age_band": "30-44",
        }

    def test_absent_metadata_yields_none(self) -> None:
        sample = Sample(id="s2", text="czesc", labels=DialectLabels(dialect="spisz"), metadata={})
        keys = record_slice_keys(sample)
        assert keys["age_band"] is None and keys["gender"] is None

    def test_metadata_aliases(self) -> None:
        sample = Sample(
            id="s3",
            text="t",
            labels=DialectLabels(family="masovian"),
            metadata={"sex": "male", "age_group": "seniors"},
        )
        keys = record_slice_keys(sample)
        assert keys["gender"] == "male" and keys["age_band"] == "seniors"


def _record(sample_id: str, *, correct: bool, gender: str) -> PredictionRecord:
    y_true = "podhale"
    y_pred = "podhale" if correct else "spisz"
    proba = (0.8, 0.2) if y_pred == "podhale" else (0.2, 0.8)
    return PredictionRecord(
        id=sample_id,
        y_true=y_true,
        y_pred=y_pred,
        proba=proba,
        source="dialektarium",
        speaker_id=f"spk-{sample_id}",
        n_chars=40,
        dialect="podhale",
        family="lesser_polish",
        region="podhale",
        gender=gender,
    )


class TestSlicingDownstream:
    def test_slice_metrics_emits_geographic_and_demographic_dimensions(self) -> None:
        records = tuple(
            _record(f"r{i}", correct=(i % 2 == 0), gender="f" if i < 6 else "m") for i in range(12)
        )
        preds = SplitPredictions(model="m", split="test", labels=_LABELS, records=records)
        dims = {m.dimension for m in slice_metrics(preds)}
        assert {"region", "family", "dialect", "gender"} <= dims
        # voivodeship/age_band were never set on any record -> no phantom dimension.
        assert "voivodeship" not in dims and "age_band" not in dims

    def test_fairness_report_flags_a_gender_gap(self) -> None:
        # Women all correct, men all wrong -> a large, real gender disparity.
        records = tuple(_record(f"f{i}", correct=True, gender="f") for i in range(8)) + tuple(
            _record(f"m{i}", correct=False, gender="m") for i in range(8)
        )
        preds = SplitPredictions(model="m", split="test", labels=_LABELS, records=records)
        gender = next(d for d in fairness_report(preds).dimensions if d.dimension == "gender")
        assert gender.best_group == "f" and gender.worst_group == "m"
        assert gender.gap == pytest.approx(1.0)


class TestOmitIfNone:
    def test_geo_demographic_keys_round_trip_and_omit_when_absent(self, tmp_path: Path) -> None:
        with_geo = PredictionRecord(
            id="a", y_true="podhale", y_pred="podhale", proba=(0.9, 0.1), dialect="podhale"
        )
        without = PredictionRecord(id="b", y_true="spisz", y_pred="spisz", proba=(0.1, 0.9))
        preds = SplitPredictions(
            model="m", split="test", labels=_LABELS, records=(with_geo, without)
        )
        path = tmp_path / "predictions.json"
        preds.save(path)
        reloaded = SplitPredictions.load(path)
        assert reloaded.records[0].dialect == "podhale"
        assert reloaded.records[1].dialect is None
        # The absent keys are omitted from the dump, not written as null.
        dumped = preds._payload()["records"]
        assert "dialect" in dumped[0] and "region" not in dumped[0]
        assert "dialect" not in dumped[1]
