"""Tests for the manifest reader, the dataset catalog/registry, and loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.core.exceptions import DataError, UnknownComponentError
from tulip.data import DATASETS, ManifestColumns, catalog, get_dataset_info, read_manifest
from tulip.data.loaders.common_voice import CommonVoiceLoader

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_REGISTRY_NAMES = [
    "bigos",
    "common_voice_pl",
    "dgp",
    "dialektarium",
    "korpus_spiski",
    "mackowce",
    "manifest",
    "nkjp",
    "spokes",
    "synthetic",
    "synthetic_audio",
]


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    import json

    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


#: A record whose transcript was empty, so the ``text`` key was simply omitted.
CONTENTLESS = {"id": "a", "dialect": "podhale", "speaker_id": "s1"}
WITH_TEXT = {"id": "b", "text": "foo bar baz lorem", "speaker_id": "s2"}
ALSO_TEXT = {"id": "c", "text": "qux quux corge", "speaker_id": "s3"}

#: Two well-formed serialised Samples, as ``save_splits`` writes them.
SPLIT_RECORDS = [
    {"id": "a", "text": "foo bar baz", "labels": {"dialect": "podhale"}, "speaker_id": "s1"},
    {"id": "b", "text": "qux quux corge", "labels": {"dialect": "silesia"}, "speaker_id": "s2"},
]


class TestJsonlHeterogeneity:
    """A JSONL manifest's records may legitimately differ in which keys they carry."""

    def test_content_column_check_is_order_independent(self, tmp_path: Path) -> None:
        """The same records must load the same way regardless of which comes first.

        The content-column check used to run against the first record only, so a
        file whose first sample had an empty (omitted) transcript was rejected
        wholesale -- while the identical records, reordered, loaded fine.
        """
        content_last = _write_jsonl(tmp_path / "a.jsonl", [CONTENTLESS, WITH_TEXT, ALSO_TEXT])
        content_first = _write_jsonl(tmp_path / "b.jsonl", [WITH_TEXT, CONTENTLESS, ALSO_TEXT])
        assert [s.id for s in read_manifest(content_last)] == ["b", "c"]
        assert [s.id for s in read_manifest(content_first)] == ["b", "c"]

    def test_a_file_with_no_content_column_anywhere_still_raises(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "none.jsonl", [CONTENTLESS, {"id": "z", "dialect": "x"}])
        with pytest.raises(DataError, match="neither a text column"):
            list(read_manifest(path))

    def test_required_columns_are_still_enforced_per_record(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "req.jsonl", [WITH_TEXT, {"id": "c", "text": "hello"}])
        with pytest.raises(DataError, match="missing required column"):
            list(read_manifest(path, columns=ManifestColumns(speaker_id="speaker_id")))


class TestReadSamplesNeverLosesLabels:
    """`read_samples` must not demote a split file to a manifest and drop its labels."""

    def test_a_valid_split_file_round_trips_its_labels(self, tmp_path: Path) -> None:
        from tulip.data.reading import read_samples

        path = _write_jsonl(tmp_path / "split.jsonl", SPLIT_RECORDS)
        assert [s.labels.dialect for s in read_samples(path)] == ["podhale", "silesia"]

    def test_a_corrupt_split_file_raises_rather_than_silently_unlabelling(
        self, tmp_path: Path
    ) -> None:
        """One bad record used to send the whole file down the manifest path.

        The nested ``labels`` object then became an unmapped column, so every
        sample came back with ``dialect=None`` -- gold labels gone, debug log only.
        """
        from tulip.data.reading import read_samples

        broken = {"id": "c", "text": None, "audio_path": None, "labels": {"dialect": "kurpie"}}
        path = _write_jsonl(tmp_path / "corrupt.jsonl", [*SPLIT_RECORDS, broken])
        with pytest.raises(DataError, match="failed validation"):
            list(read_samples(path))

    def test_a_file_mixing_split_records_and_manifest_rows_raises(self, tmp_path: Path) -> None:
        from tulip.data.reading import read_samples

        path = _write_jsonl(
            tmp_path / "mixed.jsonl",
            [*SPLIT_RECORDS, {"id": "d", "text": "x y z", "dialect": "kurpie"}],
        )
        with pytest.raises(DataError, match="mixes split-file records"):
            list(read_samples(path))

    def test_a_flat_jsonl_manifest_is_still_read_as_a_manifest(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "flat.jsonl", [{"id": "a", "text": "foo bar baz", "dialect": "podhale"}]
        )
        from tulip.data.reading import read_samples

        assert [s.labels.dialect for s in read_samples(path)] == ["podhale"]


class TestCatalogAndRegistry:
    def test_all_canonical_loaders_are_registered(self) -> None:
        assert DATASETS.names() == EXPECTED_REGISTRY_NAMES

    def test_catalog_is_tier_sorted_and_complete(self) -> None:
        infos = catalog()
        assert len(infos) == 10  # the generic manifest loader is not a corpus
        assert [info.tier for info in infos] == sorted(info.tier for info in infos)
        assert {info.name for info in infos} == set(EXPECTED_REGISTRY_NAMES) - {"manifest"}
        assert all(info.url for info in infos)

    def test_every_loader_constructs_and_reports_info(self) -> None:
        for name in DATASETS.names():
            loader = DATASETS.create(name)
            assert loader.info.name == name

    def test_unknown_dataset_suggestions(self) -> None:
        with pytest.raises(UnknownComponentError, match="dialektarium"):
            DATASETS.get("dialektarium2")
        with pytest.raises(DataError, match="unknown dataset"):
            get_dataset_info("no-such-corpus")


class TestReadManifest:
    def _write_csv(self, path: Path, rows: list[str]) -> Path:
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_csv_round_trip_with_labels_and_metadata(self, tmp_path: Path) -> None:
        manifest = self._write_csv(
            tmp_path / "manifest.csv",
            [
                "id,text,speaker_id,dialect,village,rok",
                "s1,Kaj żeś boł wczorej?,spk-a,silesia,Katowice,1978",
                "s2,Baca poseł na grań.,spk-b,podhale,Zakopane,1981",
            ],
        )
        samples = list(read_manifest(manifest, source="unit"))
        assert [s.id for s in samples] == ["s1", "s2"]
        assert samples[0].labels.dialect == "silesia"
        assert samples[0].labels.family == "silesian"  # derived by DialectLabels
        assert samples[0].labels.village == "Katowice"
        assert samples[0].metadata == {"rok": "1978"}  # unmapped columns preserved
        assert samples[0].source == "unit"

    def test_jsonl_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            '{"text": "Hej, baca się pyto.", "speaker_id": "spk-1", "dialect": "podhale"}\n',
            encoding="utf-8",
        )
        (sample,) = read_manifest(manifest)
        assert sample.labels.dialect == "podhale"
        assert sample.id  # synthesised

    def test_jsonl_required_columns_enforced_per_record(self, tmp_path: Path) -> None:
        # JSONL records may be heterogeneous, so an explicitly-required column
        # must be checked on every line, not just the first.
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            '{"text": "Pierwszy wiersz.", "gwara": "podhale"}\n'
            '{"text": "Drugi wiersz bez etykiety."}\n',
            encoding="utf-8",
        )
        with pytest.raises(DataError, match=r"manifest\.jsonl:2.*gwara"):
            list(read_manifest(manifest, columns=ManifestColumns(dialect="gwara")))

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        manifest = self._write_csv(tmp_path / "manifest.csv", ["text,speaker_id", "abc,spk-1"])
        with pytest.raises(DataError, match="missing required column"):
            list(read_manifest(manifest, columns=ManifestColumns(dialect="gwara")))

    def test_manifest_without_text_or_audio_column_raises(self, tmp_path: Path) -> None:
        manifest = self._write_csv(tmp_path / "manifest.csv", ["id,dialect", "s1,podhale"])
        with pytest.raises(DataError, match="neither a text column"):
            list(read_manifest(manifest))

    def test_surrogate_speaker_groups_by_locality(self, tmp_path: Path) -> None:
        manifest = self._write_csv(
            tmp_path / "manifest.csv",
            [
                "text,village,dialect",
                "Pierwsza wypowiedź z Chochołowa.,Chochołów,podhale",
                "Druga wypowiedź z Chochołowa.,Chochołów,podhale",
                "Wypowiedź z Katowic.,Katowice,silesia",
            ],
        )
        samples = list(read_manifest(manifest))
        assert samples[0].speaker_id == samples[1].speaker_id  # same locality groups
        assert samples[0].speaker_id != samples[2].speaker_id
        assert all(s.speaker_id and s.speaker_id.startswith("spk-") for s in samples)

    def test_relative_audio_paths_resolve_against_root(self, tmp_path: Path) -> None:
        manifest = self._write_csv(
            tmp_path / "manifest.csv",
            ["audio_path,speaker_id", "clips/a.wav,spk-1"],
        )
        (sample,) = read_manifest(manifest, audio_root=tmp_path)
        assert sample.audio_path == tmp_path / "clips" / "a.wav"

    def test_registered_loader_probes_default_manifest_names(self, tmp_path: Path) -> None:
        self._write_csv(
            tmp_path / "manifest.tsv",
            ["text\tspeaker_id\tdialect", "Kaj żeś boł?\tspk-1\tsilesia"],
        )
        loader = DATASETS.create("dialektarium")
        assert loader.is_available(tmp_path)
        (sample,) = loader.load(tmp_path)
        assert sample.source == "dialektarium"

    def test_loader_reports_unavailable_and_raises_helpfully(self, tmp_path: Path) -> None:
        loader = DATASETS.create("dgp")
        assert not loader.is_available(tmp_path)
        with pytest.raises(DataError, match=r"docs/datasets\.md"):
            list(loader.load(tmp_path))


class TestCommonVoiceLoader:
    HEADER = "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccents\tlocale"

    def _write_tsv(self, root: Path, rows: list[str]) -> None:
        (root / "validated.tsv").write_text(
            "\n".join([self.HEADER, *rows]) + "\n", encoding="utf-8"
        )

    def test_parses_official_layout(self, tmp_path: Path) -> None:
        self._write_tsv(
            tmp_path,
            [
                "abc123\tclip1.mp3\tDzień dobry państwu.\t2\t0\tthirties\tmale\t\tpl",
                "def456\tclip2.mp3\tMiło mi poznać.\t1\t0\t\t\t\tpl",
            ],
        )
        loader = CommonVoiceLoader()
        assert loader.is_available(tmp_path)
        samples = list(loader.load(tmp_path))
        assert len(samples) == 2
        assert samples[0].speaker_id == "abc123"  # client_id drives split grouping
        assert samples[0].audio_path == tmp_path / "clips" / "clip1.mp3"
        assert samples[0].labels.family == "standard"
        assert samples[0].metadata["age"] == "thirties"

    def test_accent_mapping_promotes_dialect_labels(self, tmp_path: Path) -> None:
        self._write_tsv(
            tmp_path,
            ["abc\tclip.mp3\tGodom po naszymu.\t1\t0\t\t\tśląski\tpl"],
        )
        loader = CommonVoiceLoader(accent_to_dialect={"Śląski": "silesia"})
        (sample,) = loader.load(tmp_path)
        assert sample.labels.dialect == "silesia"
        assert sample.labels.family == "silesian"

    def test_missing_columns_raise(self, tmp_path: Path) -> None:
        (tmp_path / "validated.tsv").write_text("foo\tbar\n1\t2\n", encoding="utf-8")
        with pytest.raises(DataError, match="missing expected column"):
            list(CommonVoiceLoader().load(tmp_path))

    def test_missing_file_raises_with_acquisition_hint(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match=r"commonvoice\.mozilla\.org"):
            list(CommonVoiceLoader().load(tmp_path))
