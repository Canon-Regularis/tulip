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
        self, fake_bigos_hub: SimpleNamespace, tmp_path: Path
    ) -> None:
        reports = download_datasets(None, tmp_path)
        assert [report.name for report in reports] == [info.name for info in catalog()]
        statuses = {report.name: report.status for report in reports}
        assert statuses["bigos"] is DownloadStatus.DOWNLOADED
        assert all(
            status is DownloadStatus.MANUAL for name, status in statuses.items() if name != "bigos"
        )

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
