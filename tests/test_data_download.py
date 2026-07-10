"""Tests for corpus acquisition (tulip.data.download + loader downloaders)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tulip.core.exceptions import ConfigurationError, DataError, UnknownComponentError
from tulip.data import DATASETS, DownloadStatus, catalog, download_datasets


class TestBigosDownload:
    def test_download_materialises_manifest_and_round_trips(
        self, fake_bigos_hub: SimpleNamespace, tmp_path: Path
    ) -> None:
        loader = DATASETS.create("bigos")
        assert loader.auto_downloadable

        loader.download(tmp_path / "bigos")

        assert fake_bigos_hub.load_dataset_args == (
            "michaljunczyk/pl-asr-bigos",
            None,
            "train",
            True,
        )
        # The manifest mode (default) must now work fully offline.
        samples = list(DATASETS.create("bigos").load(tmp_path / "bigos"))
        assert len(samples) == 3  # empty-text record skipped
        assert samples[0].text == "Pierwsze zdanie testowe."
        assert samples[0].speaker_id == "spk-1"
        assert samples[2].text == "Trzecie, z przecinkiem w tekście."  # CSV quoting held
        assert samples[2].speaker_id  # surrogate synthesised for missing speaker

    def test_download_respects_limit_option(
        self, fake_bigos_hub: SimpleNamespace, tmp_path: Path
    ) -> None:
        DATASETS.create("bigos").download(tmp_path / "bigos", limit=1)
        samples = list(DATASETS.create("bigos").load(tmp_path / "bigos"))
        assert len(samples) == 1

    def test_download_rejects_unknown_options(
        self, fake_bigos_hub: SimpleNamespace, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigurationError, match="unknown option"):
            DATASETS.create("bigos").download(tmp_path / "bigos", codec="flac")

    def test_empty_stream_raises_and_leaves_no_partial_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        module = ModuleType("datasets")
        module.load_dataset = lambda *args, **kwargs: iter(())
        monkeypatch.setitem(sys.modules, "datasets", module)
        with pytest.raises(DataError, match="no samples"):
            DATASETS.create("bigos").download(tmp_path / "bigos")
        assert not (tmp_path / "bigos" / "manifest.csv").exists()

    def test_gated_failure_leaves_no_partial_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Regression: _load_from_hub is a generator, so load_dataset raises on
        # FIRST ITERATION — after the CSV header was already written. The
        # header-only file must not survive to masquerade as a present corpus.
        def gated(*args, **kwargs):
            raise RuntimeError("gated dataset on the Hub. You must be authenticated")

        module = ModuleType("datasets")
        module.load_dataset = gated
        monkeypatch.setitem(sys.modules, "datasets", module)
        with pytest.raises(DataError, match="hf auth login"):
            DATASETS.create("bigos").download(tmp_path / "bigos")
        assert not (tmp_path / "bigos" / "manifest.csv").exists()


_TEI_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<teiCorpus xmlns:xi="http://www.w3.org/2001/XInclude" xmlns="http://www.tei-c.org/ns/1.0">
 <TEI>
  <text xml:id="txt_text" xml:lang="pl">
   <body xml:id="txt_body">
    <div xml:id="txt_1-div">
     <ab xml:id="txt_1.1-ab">Pierwszy akapit dokumentu.</ab>
     <ab xml:id="txt_1.2-ab">Akapit z <hi rend="italic">zagnieżdżonym</hi> tekstem.</ab>
     <ab xml:id="txt_1.3-ab">  </ab>
    </div>
   </body>
  </text>
 </TEI>
</teiCorpus>
"""

_TEI_DOC_P = _TEI_DOC.replace("<ab ", "<p ").replace("</ab>", "</p>")


def _make_nkjp_archive(path: Path) -> Path:
    """Build a miniature NKJP-1M-shaped tar.gz (real member layout)."""
    import io
    import tarfile

    def add(tar: tarfile.TarFile, name: str, content: str) -> None:
        data = content.encode("utf-8")
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with tarfile.open(path, "w:gz") as tar:
        add(tar, "./doc-A/text.xml", _TEI_DOC)
        add(tar, "./doc-A/ann_words.xml", "<ignored/>")  # annotation layers skipped
        add(tar, "./doc-B/text.xml", _TEI_DOC_P)
        add(tar, "./doc-C/text.xml", "<broken><xml")  # malformed: skipped, not fatal
    return path


class TestNkjpDownload:
    def test_download_parses_tei_into_manifest(self, tmp_path: Path) -> None:
        archive = _make_nkjp_archive(tmp_path / "mini-nkjp.tar.gz")
        loader = DATASETS.create("nkjp")
        assert loader.auto_downloadable

        loader.download(tmp_path / "nkjp", url=archive.as_uri())

        samples = list(DATASETS.create("nkjp").load(tmp_path / "nkjp"))
        texts = [sample.text for sample in samples]
        assert "Pierwszy akapit dokumentu." in texts
        assert "Akapit z zagnieżdżonym tekstem." in texts  # inline markup flattened
        assert len(samples) == 4  # blank <ab> dropped; malformed doc skipped
        assert {sample.speaker_id for sample in samples} == {"doc-A", "doc-B"}
        assert {sample.labels.family for sample in samples} == {"standard"}

    def test_download_respects_limit(self, tmp_path: Path) -> None:
        archive = _make_nkjp_archive(tmp_path / "mini-nkjp.tar.gz")
        DATASETS.create("nkjp").download(tmp_path / "nkjp", url=archive.as_uri(), limit=1)
        assert len(list(DATASETS.create("nkjp").load(tmp_path / "nkjp"))) == 1

    def test_corrupt_archive_raises_and_leaves_no_manifest(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.tar.gz"
        bogus.write_bytes(b"definitely not a tarball")
        with pytest.raises(DataError, match="tar"):
            DATASETS.create("nkjp").download(tmp_path / "nkjp", url=bogus.as_uri())
        assert not (tmp_path / "nkjp" / "manifest.csv").exists()

    def test_unknown_options_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="unknown option"):
            DATASETS.create("nkjp").download(tmp_path / "nkjp", codec="flac")


class TestCommonVoiceDownload:
    HEADER = "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccents\tlocale"

    def _mirror_tsv(self, path: Path, rows: int = 5) -> Path:
        lines = [self.HEADER] + [
            f"spk{i}\tclip{i}.mp3\tZdanie numer {i}.\t2\t0\t\t\t\tpl" for i in range(rows)
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_download_places_tsv_and_round_trips(self, tmp_path: Path) -> None:
        mirror = self._mirror_tsv(tmp_path / "mirror.tsv")
        loader = DATASETS.create("common_voice_pl")
        assert loader.auto_downloadable

        loader.download(tmp_path / "cv", url=mirror.as_uri())

        samples = list(DATASETS.create("common_voice_pl").load(tmp_path / "cv"))
        assert len(samples) == 5
        assert samples[0].speaker_id == "spk0"
        assert samples[0].labels.family == "standard"

    def test_download_limit_truncates_rows(self, tmp_path: Path) -> None:
        mirror = self._mirror_tsv(tmp_path / "mirror.tsv", rows=10)
        DATASETS.create("common_voice_pl").download(tmp_path / "cv", url=mirror.as_uri(), limit=3)
        assert len(list(DATASETS.create("common_voice_pl").load(tmp_path / "cv"))) == 3

    def test_non_cv_content_raises_and_cleans_up(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.tsv"
        bogus.write_text("<html>rate limited</html>\n", encoding="utf-8")
        with pytest.raises(DataError, match="not a Common Voice release TSV"):
            DATASETS.create("common_voice_pl").download(tmp_path / "cv", url=bogus.as_uri())
        assert not (tmp_path / "cv" / "validated.tsv").exists()

    def test_unknown_options_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="unknown option"):
            DATASETS.create("common_voice_pl").download(tmp_path / "cv", audio=True)


class TestLoaderContract:
    def test_manual_loaders_refuse_download_with_guidance(self, tmp_path: Path) -> None:
        loader = DATASETS.create("dialektarium")
        assert not loader.auto_downloadable
        with pytest.raises(DataError, match=r"dialektarium\.pl"):
            loader.download(tmp_path / "dialektarium")

    def test_every_manual_loader_documents_acquisition(self) -> None:
        for info in catalog():
            loader = DATASETS.create(info.name)
            if not loader.auto_downloadable:
                assert loader.acquisition, info.name
                assert "docs/datasets.md" in loader.acquisition, info.name


class TestDownloadDatasets:
    def test_mixed_request_reports_each_outcome(
        self, fake_bigos_hub: SimpleNamespace, tmp_path: Path
    ) -> None:
        reports = download_datasets(["bigos", "dgp"], tmp_path)
        by_name = {report.name: report for report in reports}
        assert by_name["bigos"].status is DownloadStatus.DOWNLOADED
        assert by_name["dgp"].status is DownloadStatus.MANUAL
        assert "przewodnik.tmjp.pl" in by_name["dgp"].detail
        assert by_name["dgp"].destination == tmp_path / "dgp"

    def test_present_corpus_skipped_unless_forced(
        self, fake_bigos_hub: SimpleNamespace, tmp_path: Path
    ) -> None:
        first = download_datasets(["bigos"], tmp_path)
        assert first[0].status is DownloadStatus.DOWNLOADED
        again = download_datasets(["bigos"], tmp_path)
        assert again[0].status is DownloadStatus.ALREADY_PRESENT
        forced = download_datasets(["bigos"], tmp_path, force=True)
        assert forced[0].status is DownloadStatus.DOWNLOADED

    def test_none_means_the_whole_catalog(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Stub every auto downloader: a catalog-wide run must never reach the
        # network in tests.
        from tulip.data.loaders.bigos import BigosLoader
        from tulip.data.loaders.common_voice import CommonVoiceLoader
        from tulip.data.loaders.nkjp import NkjpLoader

        def fake_download(self, root: Path, **options) -> None:
            root.mkdir(parents=True, exist_ok=True)
            (root / "downloaded.marker").write_text("ok", encoding="utf-8")

        for loader_cls in (BigosLoader, CommonVoiceLoader, NkjpLoader):
            monkeypatch.setattr(loader_cls, "download", fake_download)

        reports = download_datasets(None, tmp_path)
        assert [report.name for report in reports] == [info.name for info in catalog()]
        statuses = {report.name: report.status for report in reports}
        automatic = {"bigos", "common_voice_pl", "nkjp"}
        for name in automatic:
            assert statuses[name] is DownloadStatus.DOWNLOADED, name
        for name, status in statuses.items():
            if name not in automatic:
                assert status is DownloadStatus.MANUAL, name

    def test_unknown_corpus_raises_with_suggestions(self, tmp_path: Path) -> None:
        with pytest.raises(UnknownComponentError, match="bigos"):
            download_datasets(["bigoss"], tmp_path)

    def test_gated_hub_failure_reports_failed_and_continues(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Regression: a gated BIGOS must not abort the rest of an --all run,
        # and the failure detail must carry the authentication remediation.
        def gated(*args, **kwargs):
            raise RuntimeError(
                "Dataset 'michaljunczyk/pl-asr-bigos' is a gated dataset on the "
                "Hub. You must be authenticated to access it."
            )

        module = ModuleType("datasets")
        module.load_dataset = gated
        monkeypatch.setitem(sys.modules, "datasets", module)

        reports = download_datasets(["bigos", "dgp"], tmp_path)
        by_name = {report.name: report for report in reports}
        assert by_name["bigos"].status is DownloadStatus.FAILED
        assert "hf auth login" in by_name["bigos"].detail
        assert "huggingface.co/datasets/michaljunczyk/pl-asr-bigos" in by_name["bigos"].detail
        assert by_name["dgp"].status is DownloadStatus.MANUAL  # run continued
