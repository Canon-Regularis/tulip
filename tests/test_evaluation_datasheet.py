"""Tests for the Gebru-style dataset datasheet generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.core.types import DatasetInfo, DialectLabels, Sample
from tulip.data.splitting import DatasetSplits
from tulip.evaluation.datasheet import DatasheetSpec, datasheet, load_datasheet_spec

if TYPE_CHECKING:
    from pathlib import Path

_INFO = DatasetInfo(
    name="dialektarium",
    description="Real dialectal Polish text.",
    url="https://example.org",
    tier=1,
    tasks=("text",),
    contents=("transcribed speech",),
    label_levels=("family", "dialect"),
    license="CC-BY-4.0",
)


def _sample(
    idx: int,
    dialect: str,
    *,
    voivodeship: str | None = None,
    age: str | None = None,
    gender: str | None = None,
) -> Sample:
    metadata: dict[str, str] = {}
    if age is not None:
        metadata["age"] = age
    if gender is not None:
        metadata["gender"] = gender
    return Sample(
        id=f"s{idx}",
        text="hej baca sie pyto",
        speaker_id=f"spk{idx % 4}",
        source="dialektarium",
        labels=DialectLabels(dialect=dialect, voivodeship=voivodeship),
        metadata=metadata,
    )


def _splits(*, with_demographics: bool = True, with_out_of_taxonomy: bool = False) -> DatasetSplits:
    demo = {"age": "34", "gender": "F"} if with_demographics else {}
    train = [_sample(i, "podhale", voivodeship="malopolskie", **demo) for i in range(6)]
    if with_out_of_taxonomy:
        train.append(_sample(99, "atlantis"))  # a corpus-specific label with no centroid
    validation = [_sample(i + 10, "spisz") for i in range(3)]
    test = [_sample(i + 20, "podhale") for i in range(3)]
    return DatasetSplits(train=train, validation=validation, test=test)


_SPEC = DatasheetSpec(motivation="To benchmark Polish dialect ID.", uses="Benchmarking only.")


class TestDatasheet:
    def test_byte_stable(self) -> None:
        splits = _splits()
        assert datasheet(_INFO, splits, _SPEC) == datasheet(_INFO, splits, _SPEC)

    def test_class_distribution_at_every_level(self) -> None:
        doc = datasheet(_INFO, _splits(), _SPEC)
        assert "### Class distribution: family" in doc
        assert "### Class distribution: dialect" in doc
        assert "### Class distribution: voivodeship" in doc
        # Family auto-derives from dialect: podhale + spisz are both lesser_polish.
        assert "Lesser Polish" in doc

    def test_speaker_disjoint_counts_in_composition(self) -> None:
        doc = datasheet(_INFO, _splits(), _SPEC)
        assert "Distinct speakers" in doc and "## Composition" in doc

    def test_geographic_section_lists_only_in_taxonomy_labels(self) -> None:
        doc = datasheet(_INFO, _splits(with_out_of_taxonomy=True), _SPEC)
        geo = doc.split("## Geographic distribution", 1)[1].split("## Demographic", 1)[0]
        assert "Podhale" in geo and "49.35" in geo  # in-taxonomy centroid rendered
        assert "malopolskie" in geo and "49.90" in geo  # voivodeship centroid
        assert "atlantis" not in geo.lower()  # out-of-taxonomy: no centroid, excluded
        # ...but the out-of-taxonomy label still counts in the class distribution.
        assert "atlantis" in doc.lower()

    def test_demographic_section_present_and_absent(self) -> None:
        with_demo = datasheet(_INFO, _splits(with_demographics=True), _SPEC)
        assert "### Age band" in with_demo and "30-44" in with_demo
        assert "### Gender" in with_demo and "| f |" in with_demo  # lowercased

        without = datasheet(_INFO, _splits(with_demographics=False), _SPEC)
        assert "No demographic metadata" in without

    def test_absent_prose_fields_degrade(self) -> None:
        doc = datasheet(_INFO, _splits(), DatasheetSpec())  # every prose field blank
        assert "_Not documented._" in doc
        assert "## Preprocessing" in doc  # section still rendered

    def test_license_in_distribution(self) -> None:
        doc = datasheet(_INFO, _splits(), _SPEC)
        assert "CC-BY-4.0" in doc


class TestLoadSpec:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "spec.yaml"
        path.write_text("motivation: hello\nuses: research\n", encoding="utf-8")
        spec = load_datasheet_spec(path)
        assert spec.motivation == "hello" and spec.uses == "research"

    def test_empty_file_is_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_datasheet_spec(path) == DatasheetSpec()
