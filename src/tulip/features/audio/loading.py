"""Shared audio decoding and resampling for every audio consumer in tulip.

Every feature extractor and speech model funnels file access through
:func:`load_audio`, so decoding, mono downmixing, and resampling behave
identically across the toolkit. soundfile (libsndfile) is the primary decoder
because it is fast and dependency-light; librosa is the fallback for container
formats libsndfile cannot read. Both are optional dependencies of the
``audio`` extra and are imported lazily, so importing this module never
requires them.

Decoded audio is memoised in a small bounded LRU cache: a
:class:`~sklearn.pipeline.FeatureUnion` of N audio extractors (and
single-file prediction through ``DialectClassifier``) would otherwise decode
the same file N times. Cached arrays are returned **read-only** so one
consumer cannot corrupt another's view; copy before mutating. With the
default cache size (32) and 30 s clips at 16 kHz the cache tops out around
60 MB.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.utils import optional
from tulip.utils.logging import get_logger

logger = get_logger(__name__)

#: Target sample rate shared by all audio features (matches wav2vec2-style models).
DEFAULT_SAMPLE_RATE = 16_000

#: Files kept in the decoded-audio LRU cache (see the module docstring).
DECODE_CACHE_SIZE = 32

__all__ = [
    "DECODE_CACHE_SIZE",
    "DEFAULT_SAMPLE_RATE",
    "clear_audio_cache",
    "load_audio",
    "resample",
]


def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D signal with a polyphase filter (scipy, no optional deps).

    Args:
        signal: 1-D audio signal.
        orig_sr: Sample rate of ``signal`` in Hz.
        target_sr: Desired sample rate in Hz.

    Returns:
        The resampled signal as contiguous float32.

    Raises:
        ConfigurationError: If either sample rate is not positive.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ConfigurationError(
            f"sample rates must be positive, got orig_sr={orig_sr}, target_sr={target_sr}"
        )
    data = np.asarray(signal, dtype=np.float64).ravel()
    if orig_sr == target_sr or data.size == 0:
        return np.ascontiguousarray(data, dtype=np.float32)
    divisor = math.gcd(int(orig_sr), int(target_sr))
    resampled = resample_poly(data, int(target_sr) // divisor, int(orig_sr) // divisor)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def load_audio(path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file to a mono float32 signal at ``sample_rate`` Hz.

    soundfile is tried first (libsndfile decoding + scipy polyphase
    resampling); ``librosa.load`` is the fallback for formats libsndfile
    cannot decode (e.g. mp3 on some builds). Results are memoised in a
    bounded LRU cache keyed by path, rate, and file mtime/size, so feature
    unions and repeated predictions do not re-decode the same file.

    Args:
        path: Audio file to decode.
        sample_rate: Target sample rate in Hz.

    Returns:
        1-D contiguous float32 array of samples at ``sample_rate`` Hz. The
        array is **read-only** (it may be shared with other cache consumers);
        ``signal.copy()`` before any in-place modification.

    Raises:
        DataError: If the file does not exist or cannot be decoded.
        ConfigurationError: If ``sample_rate`` is not positive.
        MissingDependencyError: If neither soundfile nor librosa is installed
            (install the ``audio`` extra).
    """
    if sample_rate <= 0:
        raise ConfigurationError(f"target sample rate must be positive, got {sample_rate}")
    audio_path = Path(path)
    try:
        stat = audio_path.stat()
    except OSError as exc:
        raise DataError(f"audio file not found: {audio_path}") from exc
    return _cached_decode(str(audio_path), sample_rate, stat.st_mtime_ns, stat.st_size)


def clear_audio_cache() -> None:
    """Drop all cached decoded audio (e.g. between benchmark runs)."""
    _cached_decode.cache_clear()


@lru_cache(maxsize=DECODE_CACHE_SIZE)
def _cached_decode(path: str, sample_rate: int, mtime_ns: int, size: int) -> np.ndarray:
    """Decode once per (path, rate, mtime, size); the stat fields bust staleness."""
    del mtime_ns, size  # cache-key components only
    signal = _decode(Path(path), sample_rate)
    signal.flags.writeable = False  # shared across cache consumers
    return signal


def _decode(audio_path: Path, sample_rate: int) -> np.ndarray:
    """Decode with soundfile, falling back to librosa for exotic containers."""
    if optional.is_available("soundfile"):
        try:
            return _load_with_soundfile(audio_path, sample_rate)
        except Exception as exc:  # undecodable by libsndfile; librosa may still manage
            logger.debug("soundfile could not decode %s (%s); trying librosa", audio_path, exc)

    librosa = optional.optional_import(
        "librosa", extra="audio", purpose="audio decoding and resampling"
    )
    try:
        signal, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as exc:
        raise DataError(f"could not decode audio file {audio_path}: {exc}") from exc
    return np.ascontiguousarray(signal, dtype=np.float32)


def _load_with_soundfile(path: Path, sample_rate: int) -> np.ndarray:
    """Decode ``path`` with soundfile and resample to ``sample_rate`` Hz mono."""
    soundfile = optional.optional_import("soundfile", extra="audio", purpose="audio decoding")
    data, native_sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return resample(mono, int(native_sr), sample_rate)
