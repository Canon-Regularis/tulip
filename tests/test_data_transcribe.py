"""Tests for the transcription bridge (fully offline; the ASR engine is faked)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import block_imports
from tulip.core.exceptions import DataError, MissingDependencyError
from tulip.core.types import DialectLabels, Sample
from tulip.data.reading import read_samples
from tulip.data.transcribe import (
    TranscribeConfig,
    TranscriptCache,
    transcribe_samples,
    write_transcribed_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FakeAsr:
    """Counts calls and returns a transcript derived from the file name."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, path: Path) -> str:
        self.calls += 1
        return f"transkrypcja {path.stem}"


def _clip(tmp_path: Path, name: str, payload: bytes = b"RIFFfake") -> Path:
    path = tmp_path / f"{name}.wav"
    path.write_bytes(payload)
    return path


def _sample(sample_id: str, audio: Path | None, dialect: str = "podhale") -> Sample:
    return Sample(
        id=sample_id,
        text=None if audio is not None else "tekst",
        audio_path=audio,
        speaker_id=f"spk-{sample_id}",
        labels=DialectLabels(dialect=dialect),
        source="test",
    )


def test_transcribes_audio_samples_and_keeps_labels(tmp_path: Path) -> None:
    asr = _FakeAsr()
    samples = [_sample("a", _clip(tmp_path, "a")), _sample("b", _clip(tmp_path, "b", b"RIFFother"))]

    out = transcribe_samples(samples, asr=asr)

    assert [s.text for s in out] == ["transkrypcja a", "transkrypcja b"]
    assert asr.calls == 2
    assert all(s.audio_path is not None for s in out)  # both modalities kept
    assert all(s.labels.dialect == "podhale" for s in out)
    assert all(s.metadata["transcribed_by"] for s in out)
    assert all(s.metadata["transcription_language"] == "pl" for s in out)


def test_samples_without_audio_are_skipped(tmp_path: Path) -> None:
    asr = _FakeAsr()
    out = transcribe_samples([_sample("t", None), _sample("a", _clip(tmp_path, "a"))], asr=asr)
    assert len(out) == 1
    assert out[0].id == "a"


def test_cache_makes_the_second_run_free(tmp_path: Path) -> None:
    config = TranscribeConfig(cache_dir=tmp_path / "cache")
    samples = [_sample("a", _clip(tmp_path, "a"))]

    first = _FakeAsr()
    transcribe_samples(samples, config, asr=first)
    assert first.calls == 1

    second = _FakeAsr()
    out = transcribe_samples(samples, config, asr=second)
    assert second.calls == 0  # served from the on-disk cache
    assert out[0].text == "transkrypcja a"


def test_cache_key_changes_with_audio_checkpoint_and_language() -> None:
    base = TranscriptCache.key(b"clip", checkpoint="c1", language="pl")
    assert TranscriptCache.key(b"other", checkpoint="c1", language="pl") != base
    assert TranscriptCache.key(b"clip", checkpoint="c2", language="pl") != base
    assert TranscriptCache.key(b"clip", checkpoint="c1", language="de") != base


def test_manifest_round_trips_through_read_samples(tmp_path: Path) -> None:
    asr = _FakeAsr()
    out = transcribe_samples([_sample("a", _clip(tmp_path, "a"))], asr=asr)
    manifest = write_transcribed_manifest(out, tmp_path / "corpus")

    loaded = list(read_samples(manifest))
    assert len(loaded) == 1
    assert loaded[0].text == "transkrypcja a"
    assert loaded[0].labels.dialect == "podhale"
    assert loaded[0].audio_path is not None
    assert loaded[0].speaker_id == "spk-a"


def test_missing_audio_file_raises_cleanly(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="cannot read audio"):
        transcribe_samples([_sample("a", tmp_path / "missing.wav")], asr=_FakeAsr())


def test_default_engine_needs_the_speech_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    block_imports(monkeypatch, "transformers")
    with pytest.raises(MissingDependencyError, match=r"tulip-dialect\[speech\]"):
        transcribe_samples([_sample("a", _clip(tmp_path, "a"))])


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        TranscribeConfig(bogus=1)  # type: ignore[call-arg]


def test_cli_data_transcribe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from tulip.cli.app import app

    # A tiny audio corpus as a samples JSONL.
    corpus = tmp_path / "audio.jsonl"
    samples = [_sample("a", _clip(tmp_path, "a")), _sample("b", _clip(tmp_path, "b"))]
    corpus.write_text("\n".join(s.model_dump_json() for s in samples) + "\n", encoding="utf-8")
    monkeypatch.setattr("tulip.data.transcribe._build_whisper_asr", lambda config: _FakeAsr())

    result = CliRunner().invoke(
        app,
        ["data", "transcribe", str(corpus), "--out", str(tmp_path / "out"), "--limit", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "transcribed corpus" in result.output
    loaded = list(read_samples(tmp_path / "out" / "manifest.jsonl"))
    assert len(loaded) == 1
    assert loaded[0].text == "transkrypcja a"
