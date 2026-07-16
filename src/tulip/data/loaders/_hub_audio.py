"""Materialise Hugging Face ``datasets`` audio values as local clip files.

The BIGOS and Common Voice loaders both stream Hub records whose ``audio``
column is a ``datasets`` Audio value. They request that column **without
decoding** (``datasets.Audio(decode=False)``), so each value carries the
original encoded file as raw ``bytes``; this writes those bytes verbatim, which
means audio fetch needs no decode backend (no ``soundfile``/``torchcodec``) and
never re-encodes the clip. As a fallback, a value that arrives already decoded
(a float ``array`` plus its ``sampling_rate``) is written as int16 PCM WAV with
the standard-library :mod:`wave` module, mirroring the synthetic-audio writer.
Reading the clips back (in the features) still needs the ``audio`` extra.
"""

from __future__ import annotations

import wave
from collections.abc import Mapping
from pathlib import Path

from tulip.core.exceptions import DataError

__all__ = ["CLIPS_DIR", "write_hub_clip"]

#: Sub-directory of a corpus root that fetched clips are written under. Both
#: loaders record audio paths relative to the root as ``clips/<name>`` so a
#: materialised corpus is relocatable, matching the manifest reader's contract.
CLIPS_DIR = "clips"

#: Extensions treated as an audio container when sanitising a clip stem, so a
#: sample id that already ends in ``.wav`` does not yield ``id.wav.wav``.
_AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"})

_INT16_MAX = 32_767
_INT16_MIN = -32_768


def write_hub_clip(audio: object, clips_dir: Path, stem: str) -> str:
    """Write one ``datasets`` Audio value under ``clips_dir``; return its file name.

    Args:
        audio: The record's ``audio`` column: a mapping carrying either
            non-empty ``bytes`` (the original encoded file, the normal case
            when the loader disabled decoding) or a decoded ``array`` with its
            ``sampling_rate`` (the fallback).
        clips_dir: Directory the clip is written to (created if absent).
        stem: File-name stem for the clip; the caller is responsible for
            passing a stem unique within the run (both loaders prefix the
            stream index). It is still sanitised to a filesystem-safe base name.

    Returns:
        The written file's name relative to ``clips_dir`` (no directory part).
        The raw-bytes path preserves the source extension (so an ``.mp3`` stays
        an ``.mp3``), defaulting to ``.wav`` only when the source has none.

    Raises:
        DataError: if ``audio`` is not an Audio mapping, or carries neither
            usable bytes nor a decoded array.
    """
    if not isinstance(audio, Mapping):
        raise DataError(
            "audio fetch expected a datasets Audio mapping (with 'bytes' or "
            f"'array'), got {type(audio).__name__}; is this an audio column?"
        )
    clips_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_stem(stem)

    raw = audio.get("bytes")
    if raw:
        # Preserve the source container (its bytes are unchanged): the real
        # extension is more honest than forcing .wav, and content-sniffing
        # decoders read it regardless. Default to .wav only for a path with none.
        suffix = Path(str(audio.get("path") or "")).suffix or ".wav"
        name = f"{safe}{suffix}"
        (clips_dir / name).write_bytes(bytes(raw))
        return name

    array = audio.get("array")
    sampling_rate = audio.get("sampling_rate")
    if array is not None and sampling_rate:
        name = f"{safe}.wav"
        _write_pcm_wav(clips_dir / name, array, int(sampling_rate))
        return name

    raise DataError(
        "audio fetch got a record whose audio has neither 'bytes' nor a decoded "
        "'array'; the Hub may serve this column without audio payloads"
    )


def _safe_stem(stem: str) -> str:
    """A filesystem-safe clip stem: drop any audio extension and odd characters."""
    candidate = Path(stem)
    base = candidate.stem if candidate.suffix.lower() in _AUDIO_SUFFIXES else stem
    cleaned = "".join(char if (char.isalnum() or char in "._-") else "_" for char in base)
    return cleaned or "clip"


def _write_pcm_wav(path: Path, array: object, sampling_rate: int) -> None:
    """Quantise a decoded float array to mono int16 PCM and write a WAV.

    The decoded ``datasets`` array is float in ``[-1, 1]``; it is scaled to
    int16 with an explicit little-endian dtype so the bytes are platform
    independent, mirroring the synthetic-audio writer. A multi-channel array is
    downmixed to mono (mean across channels) rather than flattened, so a stereo
    decode is not silently interleaved into a doubled-length clip. numpy is
    imported lazily so ``import tulip.data`` stays light for the text pipelines.
    """
    import numpy as np

    signal = np.asarray(array, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    signal = signal.ravel()
    quantised = np.clip(np.round(signal * _INT16_MAX), _INT16_MIN, _INT16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sampling_rate)
        handle.writeframes(quantised.tobytes())
