"""Tests for the manifest reader, the dataset catalog/registry, and loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from tulip.core.exceptions import DataError, UnknownComponentError
from tulip.data import DATASETS, ManifestColumns, catalog, get_dataset_info, read_manifest
from tulip.data.loaders.common_voice import CommonVoiceLoader

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
]


class TestCatalogAndRegistry:
    def test_all_canonical_loaders_are_registered(self) -> None:
        assert DATASETS.names() == EXPECTED_REGISTRY_NAMES

    def test_catalog_is_tier_sorted_and_complete(self) -> None:
        infos = catalog()
        assert len(infos) == 8  # the generic manifest loader is not a corpus
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
