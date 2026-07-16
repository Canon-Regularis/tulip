"""Tests for audio fetch: materialising Hub clips for bigos and common_voice_pl.

Every test runs offline: the ``datasets`` library is stubbed with in-memory
records. Real usage disables decoding (``Audio(decode=False)``), so a streamed
record carries the clip's original encoded ``bytes``; the fakes here mirror that
by carrying ``bytes`` too, and the loaders write them verbatim, needing no audio
extra. The decoded-``array`` fallback is exercised directly against
``write_hub_clip``.
"""

from __future__ import annotations

import sys
import wave
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from conftest import block_imports
from tulip.core.exceptions import ConfigurationError, DataError, MissingDependencyError
from tulip.data import DATASETS
from tulip.data.loaders._hub_audio import write_hub_clip

if TYPE_CHECKING:
    from pathlib import Path

#: BIGOS-shaped Hub records carrying the clip's original bytes (decode disabled).
BIGOS_AUDIO_RECORDS: list[dict[str, object]] = [
    {
        "ref_orig": "Pierwsze audio.",
        "speaker_id": "spk-1",
        "dataset": "sub-a",
        "audioname": "clip0.wav",
        "audio": {"path": "clip0.wav", "bytes": b"RIFF-clip0-fake"},
    },
    {
        "ref_orig": "Drugie audio.",
        "speaker_id": "spk-2",
        "dataset": "sub-a",
        "audioname": "clip1.wav",
        "audio": {"path": "clip1.wav", "bytes": b"RIFF-clip1-fake"},
    },
]

#: Common-Voice-shaped Hub records: original mp3 bytes plus transcript and accent.
CV_AUDIO_RECORDS: list[dict[str, object]] = [
    {
        "client_id": "cv-spk-1",
        "sentence": "Pierwsze zdanie.",
        "path": "common_voice_pl_1.mp3",
        "accents": "Małopolska",
        "audio": {"path": "common_voice_pl_1.mp3", "bytes": b"mp3-1"},
    },
    {
        "client_id": "",  # no speaker: skipped (splitting needs a speaker id)
        "sentence": "Bez mówcy.",
        "path": "common_voice_pl_2.mp3",
        "accents": "",
        "audio": {"path": "common_voice_pl_2.mp3", "bytes": b"mp3-2"},
    },
    {
        "client_id": "cv-spk-2",
        "sentence": "Trzecie zdanie.",
        "path": "common_voice_pl_3.mp3",
        "accents": "",
        "audio": {"path": "common_voice_pl_3.mp3", "bytes": b"mp3-3"},
    },
]


class _FakeHubStream:
    """A minimal streaming ``IterableDataset`` stand-in.

    Records the ``cast_column`` request (so a test can assert decoding was
    disabled) and yields the fixed records.
    """

    def __init__(self, records: list[dict[str, object]], calls: SimpleNamespace) -> None:
        self._records = records
        self._calls = calls

    def cast_column(self, name: str, feature: object) -> _FakeHubStream:
        self._calls.cast = (name, feature)
        return self

    def __iter__(self):
        return iter(self._records)


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch, records: list[dict[str, object]]
) -> SimpleNamespace:
    """Stub the ``datasets`` library so ``load_dataset`` streams ``records``."""
    calls = SimpleNamespace(args=None, cast=None)

    def load_dataset(name, config=None, *, split, streaming):
        calls.args = (name, config, split, streaming)
        return _FakeHubStream(records, calls)

    module = ModuleType("datasets")
    module.load_dataset = load_dataset
    module.Audio = lambda **kwargs: ("Audio", kwargs)  # feature stub the cast records
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


class TestWriteHubClip:
    def test_raw_bytes_are_passed_through_with_their_suffix(self, tmp_path: Path) -> None:
        payload = b"FLACblob"
        name = write_hub_clip({"bytes": payload, "path": "orig.flac"}, tmp_path, "sample-2")
        assert name == "sample-2.flac"
        assert (tmp_path / name).read_bytes() == payload

    def test_byte_suffix_is_preserved_even_when_unusual(self, tmp_path: Path) -> None:
        # The bytes are unchanged, so keeping the real container is more honest
        # than forcing .wav; a content-sniffing decoder reads it regardless.
        name = write_hub_clip({"bytes": b"blob", "path": "orig.weird"}, tmp_path, "s")
        assert name == "s.weird"

    def test_extensionless_bytes_default_to_wav(self, tmp_path: Path) -> None:
        name = write_hub_clip({"bytes": b"blob", "path": "no_extension"}, tmp_path, "s")
        assert name == "s.wav"

    def test_decoded_array_fallback_is_written_as_readable_wav(self, tmp_path: Path) -> None:
        audio = {"array": [0.0, 0.5, -0.5], "sampling_rate": 16_000, "path": "x.wav"}
        name = write_hub_clip(audio, tmp_path, "sample-1")
        assert name == "sample-1.wav"
        with wave.open(str(tmp_path / name)) as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 16_000
            assert handle.getnframes() == 3

    def test_multichannel_array_is_downmixed_not_interleaved(self, tmp_path: Path) -> None:
        # A stereo (n, 2) decode must yield n mono frames, not 2n interleaved.
        audio = {"array": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], "sampling_rate": 8_000}
        name = write_hub_clip(audio, tmp_path, "stereo")
        with wave.open(str(tmp_path / name)) as handle:
            assert handle.getnchannels() == 1
            assert handle.getnframes() == 3

    def test_stem_is_sanitised(self, tmp_path: Path) -> None:
        # A stem ending in an audio extension with odd characters must not yield
        # "id.wav.wav" or a filesystem-hostile name.
        name = write_hub_clip(
            {"array": [0.0], "sampling_rate": 8_000, "path": "a.wav"}, tmp_path, "clip 1!.wav"
        )
        assert name == "clip_1_.wav"

    def test_non_mapping_audio_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="datasets Audio mapping"):
            write_hub_clip(None, tmp_path, "s")

    def test_payloadless_audio_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="neither 'bytes' nor a decoded"):
            write_hub_clip({"path": "x.wav"}, tmp_path, "s")


class TestBigosAudioFetch:
    def test_audio_download_disables_decoding(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _install_fake_hub(monkeypatch, BIGOS_AUDIO_RECORDS)
        DATASETS.create("bigos").download(tmp_path / "bigos", audio=True)
        # The crux: the clip column is fetched without decoding, so no audio
        # backend (soundfile/torchcodec) is needed to materialise clips.
        assert calls.cast == ("audio", ("Audio", {"decode": False}))

    def test_audio_download_writes_clips_and_round_trips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_hub(monkeypatch, BIGOS_AUDIO_RECORDS)
        root = tmp_path / "bigos"

        DATASETS.create("bigos").download(root, audio=True)

        clips = sorted((root / "clips").glob("*.wav"))
        assert len(clips) == 2
        samples = list(DATASETS.create("bigos").load(root))
        assert len(samples) == 2
        assert samples[0].text == "Pierwsze audio."
        assert samples[0].audio_path is not None
        assert samples[0].audio_path.is_file()  # the manifest path resolves to a real clip
        assert samples[0].audio_path.parent == root / "clips"
        assert samples[0].audio_path.read_bytes() == b"RIFF-clip0-fake"  # original bytes verbatim

    def test_text_mode_writes_no_audio_column_or_clips(
        self, fake_bigos_hub: SimpleNamespace, tmp_path: Path
    ) -> None:
        root = tmp_path / "bigos"
        DATASETS.create("bigos").download(root)  # default: text only
        assert not (root / "clips").exists()
        header = (root / "manifest.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "audio_path" not in header

    def test_audio_download_respects_limit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_hub(monkeypatch, BIGOS_AUDIO_RECORDS)
        root = tmp_path / "bigos"
        DATASETS.create("bigos").download(root, audio=True, limit=1)
        assert len(list((root / "clips").glob("*.wav"))) == 1


class TestCommonVoiceAudioFetch:
    def test_audio_download_disables_decoding(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = _install_fake_hub(monkeypatch, CV_AUDIO_RECORDS)
        DATASETS.create("common_voice_pl").download(tmp_path / "cv", audio=True, limit=1)
        assert calls.args == ("fsicoli/common_voice_17_0", "pl", "train", True)
        assert calls.cast == ("audio", ("Audio", {"decode": False}))

    def test_audio_download_writes_clips_text_and_labels(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_hub(monkeypatch, CV_AUDIO_RECORDS)
        root = tmp_path / "cv"
        loader = DATASETS.create("common_voice_pl", accent_to_dialect={"małopolska": "podhale"})

        loader.download(root, audio=True, limit=10)

        # Original mp3 bytes are written verbatim under their real extension.
        assert len(list((root / "clips").glob("*.mp3"))) == 2
        samples = list(
            DATASETS.create("common_voice_pl", accent_to_dialect={"małopolska": "podhale"}).load(
                root
            )
        )
        # The speaker-less middle record is dropped, leaving two.
        assert [s.speaker_id for s in samples] == ["cv-spk-1", "cv-spk-2"]
        assert samples[0].text == "Pierwsze zdanie."
        assert samples[0].audio_path is not None and samples[0].audio_path.is_file()
        assert samples[0].audio_path.read_bytes() == b"mp3-1"
        assert samples[0].labels.dialect == "podhale"  # mapped accent survives the round trip
        assert samples[1].labels.dialect is None  # blank accent stays standard

    def test_audio_download_respects_limit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_hub(monkeypatch, CV_AUDIO_RECORDS)
        root = tmp_path / "cv"
        DATASETS.create("common_voice_pl").download(root, audio=True, limit=1)
        # limit counts written rows, and the first record has a speaker.
        assert len(list((root / "clips").glob("*.mp3"))) == 1

    def test_empty_audio_stream_leaves_no_partial_tsv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_hub(monkeypatch, [])
        root = tmp_path / "cv"
        with pytest.raises(DataError, match="produced no samples"):
            DATASETS.create("common_voice_pl").download(root, audio=True)
        assert not (root / "validated.tsv").exists()

    def test_audio_download_rejects_unknown_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_hub(monkeypatch, CV_AUDIO_RECORDS)
        with pytest.raises(ConfigurationError, match="unknown option"):
            DATASETS.create("common_voice_pl").download(tmp_path / "cv", audio=True, codec="opus")

    def test_audio_download_needs_the_hf_extra(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        block_imports(monkeypatch, "datasets")
        with pytest.raises(MissingDependencyError, match=r"tulip-dialect\[hf\]"):
            DATASETS.create("common_voice_pl").download(tmp_path / "cv", audio=True)


class TestAudioFetchFailureCleanup:
    def test_midstream_error_removes_tsv_and_clips_as_a_tulip_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A record whose audio has no payload raises mid-stream (after the first
        # good clip was written). The partial TSV and the orphan clip must both
        # be cleaned, and a bare library error must surface as a TulipError so
        # download_datasets' per-corpus handler catches it.
        records = [
            dict(CV_AUDIO_RECORDS[0]),
            {
                "client_id": "cv-spk-9",
                "sentence": "x",
                "path": "bad.mp3",
                "audio": {"path": "b.mp3"},
            },
        ]
        _install_fake_hub(monkeypatch, records)
        root = tmp_path / "cv"
        with pytest.raises(DataError):
            DATASETS.create("common_voice_pl").download(root, audio=True)
        assert not (root / "validated.tsv").exists()
        assert list((root / "clips").glob("*")) == []  # the one good clip was cleaned up

    def test_bigos_midstream_error_removes_manifest_and_clips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Same guarantee for bigos: a payloadless audio record after a good one
        # leaves neither a partial manifest nor an orphan clip behind.
        records = [
            dict(BIGOS_AUDIO_RECORDS[0]),
            {
                "ref_orig": "Trzecie audio.",
                "speaker_id": "spk-9",
                "dataset": "sub-a",
                "audioname": "bad.wav",
                "audio": {"path": "bad.wav"},  # no bytes and no array
            },
        ]
        _install_fake_hub(monkeypatch, records)
        root = tmp_path / "bigos"
        with pytest.raises(DataError):
            DATASETS.create("bigos").download(root, audio=True)
        assert not (root / "manifest.csv").exists()
        assert list((root / "clips").glob("*")) == []


class TestDownloadDatasetsAudioCapability:
    def test_audio_dropped_for_loaders_without_support(self) -> None:
        from tulip.data.download import _options_for

        class NoAudio:
            pass

        class WithAudio:
            supports_audio_fetch = True

        assert _options_for(NoAudio(), {"audio": True, "limit": 5}) == {"limit": 5}
        # Even an explicit audio=False is dropped for a non-supporting loader, so
        # a public-API caller cannot trip its unknown-option guard.
        assert _options_for(NoAudio(), {"audio": False, "limit": 5}) == {"limit": 5}
        assert _options_for(WithAudio(), {"audio": True, "limit": 5}) == {
            "audio": True,
            "limit": 5,
        }

    def test_original_options_are_not_mutated(self) -> None:
        from tulip.data.download import _options_for

        class NoAudio:
            pass

        shared = {"audio": True, "limit": 3}
        _options_for(NoAudio(), shared)
        assert shared == {"audio": True, "limit": 3}  # the per-loader copy is isolated

    def test_batch_download_materialises_audio_for_supporting_loader(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tulip.data import DownloadStatus, download_datasets

        _install_fake_hub(monkeypatch, BIGOS_AUDIO_RECORDS)
        reports = download_datasets(["bigos"], tmp_path, options={"audio": True})
        assert reports[0].status is DownloadStatus.DOWNLOADED
        assert len(list((tmp_path / "bigos" / "clips").glob("*.wav"))) == 2


class TestCliDownloadAudio:
    def test_download_audio_flag_writes_clips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from typer.testing import CliRunner

        from tulip.cli.app import app

        _install_fake_hub(monkeypatch, BIGOS_AUDIO_RECORDS)
        result = CliRunner().invoke(
            app, ["data", "download", "bigos", "--audio", "--root", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        assert len(list((tmp_path / "bigos" / "clips").glob("*.wav"))) == 2
