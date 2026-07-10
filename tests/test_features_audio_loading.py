"""Tests for the canonical shared audio loader (tulip.features.audio.loading)."""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.features.audio.loading import clear_audio_cache, load_audio, resample

if TYPE_CHECKING:
    from pathlib import Path


def _write_wav(path: Path, *, seconds: float, framerate: int, channels: int = 1) -> Path:
    frames = int(seconds * framerate)
    samples = (np.sin(np.linspace(0, 440 * np.pi, frames)) * 12_000).astype(np.int16)
    if channels == 2:
        samples = np.column_stack([samples, samples]).ravel()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(framerate)
        handle.writeframes(samples.tobytes())
    return path


class TestResample:
    def test_identity_at_same_rate(self) -> None:
        signal = np.linspace(-1, 1, 50, dtype=np.float32)
        assert resample(signal, 16_000, 16_000) == pytest.approx(signal)

    def test_doubles_length_8k_to_16k(self) -> None:
        resampled = resample(np.ones(8_000, dtype=np.float32), 8_000, 16_000)
        assert resampled.shape == (16_000,)
        assert resampled.dtype == np.float32
        # interior of a constant signal stays ~constant after polyphase filtering
        assert resampled[100:-100] == pytest.approx(np.ones(15_800), abs=1e-3)

    def test_rejects_nonpositive_rates(self) -> None:
        with pytest.raises(ConfigurationError):
            resample(np.zeros(10, dtype=np.float32), 0, 16_000)


class TestLoadAudio:
    def test_missing_file_raises_data_error(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="not found"):
            load_audio(tmp_path / "missing.wav")

    def test_nonpositive_rate_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="positive"):
            load_audio(tmp_path / "whatever.wav", sample_rate=0)

    def test_decodes_and_resamples(self, tmp_path: Path) -> None:
        pytest.importorskip("soundfile")
        clip = _write_wav(tmp_path / "clip.wav", seconds=1.0, framerate=8_000)
        assert load_audio(clip, sample_rate=8_000).shape == (8_000,)
        assert load_audio(clip, sample_rate=16_000).shape == (16_000,)

    def test_downmixes_stereo_to_mono(self, tmp_path: Path) -> None:
        pytest.importorskip("soundfile")
        clip = _write_wav(tmp_path / "stereo.wav", seconds=0.5, framerate=16_000, channels=2)
        signal = load_audio(clip, sample_rate=16_000)
        assert signal.ndim == 1
        assert signal.shape == (8_000,)


class TestDecodeCache:
    def test_repeat_loads_hit_the_cache_and_are_read_only(self, tmp_path: Path) -> None:
        pytest.importorskip("soundfile")
        clear_audio_cache()
        clip = _write_wav(tmp_path / "clip.wav", seconds=0.25, framerate=16_000)
        first = load_audio(clip)
        second = load_audio(clip)
        assert first is second  # decoded once, shared thereafter
        assert not first.flags.writeable  # shared arrays must be immutable
        with pytest.raises(ValueError):
            first[0] = 1.0

    def test_different_rates_are_distinct_entries(self, tmp_path: Path) -> None:
        pytest.importorskip("soundfile")
        clear_audio_cache()
        clip = _write_wav(tmp_path / "clip.wav", seconds=0.25, framerate=16_000)
        assert (
            load_audio(clip, sample_rate=16_000).shape != load_audio(clip, sample_rate=8_000).shape
        )

    def test_file_change_busts_the_cache(self, tmp_path: Path) -> None:
        pytest.importorskip("soundfile")
        clear_audio_cache()
        clip = _write_wav(tmp_path / "clip.wav", seconds=0.25, framerate=16_000)
        before = load_audio(clip)
        _write_wav(clip, seconds=0.5, framerate=16_000)  # different size -> new key
        after = load_audio(clip)
        assert after.shape != before.shape
