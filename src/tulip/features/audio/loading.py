"""Shared audio decoding and resampling for all audio feature extractors.

Every extractor funnels file access through :func:`load_audio` so decoding,
mono downmixing, and resampling behave identically across features. soundfile
(libsndfile) is the primary decoder because it is fast and dependency-light;
librosa is the fallback for container formats libsndfile cannot read. Both are
optional dependencies of the ``audio`` extra and are imported lazily, so
importing this module never requires them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.utils import optional
from tulip.utils.logging import get_logger

logger = get_logger(__name__)

#: Target sample rate shared by all audio features (matches wav2vec2-style models).
DEFAULT_SAMPLE_RATE = 16_000

__all__ = ["DEFAULT_SAMPLE_RATE", "load_audio", "resample"]


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


def load_audio(path: str | Path, sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file to a mono float32 signal at ``sr`` Hz.

    soundfile is tried first (libsndfile decoding + scipy polyphase
    resampling); ``librosa.load`` is the fallback for formats libsndfile
    cannot decode (e.g. mp3 on some builds). Arguments are hashable so callers
    that repeatedly extract features from the same files can wrap this
    function in ``functools.lru_cache``.

    Args:
        path: Audio file to decode.
        sr: Target sample rate in Hz.

    Returns:
        1-D contiguous float32 array of samples at ``sr`` Hz.

    Raises:
        DataError: If the file does not exist or cannot be decoded.
        ConfigurationError: If ``sr`` is not positive.
        MissingDependencyError: If neither soundfile nor librosa is installed
            (install the ``audio`` extra).
    """
    if sr <= 0:
        raise ConfigurationError(f"target sample rate must be positive, got {sr}")
    audio_path = Path(path)
    if not audio_path.is_file():
        raise DataError(f"audio file not found: {audio_path}")

    if optional.is_available("soundfile"):
        try:
            return _load_with_soundfile(audio_path, sr)
        except Exception as exc:  # undecodable by libsndfile; librosa may still manage
            logger.debug("soundfile could not decode %s (%s); trying librosa", audio_path, exc)

    librosa = optional.optional_import(
        "librosa", extra="audio", purpose="audio decoding and resampling"
    )
    try:
        signal, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as exc:
        raise DataError(f"could not decode audio file {audio_path}: {exc}") from exc
    return np.ascontiguousarray(signal, dtype=np.float32)


def _load_with_soundfile(path: Path, sr: int) -> np.ndarray:
    """Decode ``path`` with soundfile and resample to ``sr`` Hz mono."""
    soundfile = optional.optional_import("soundfile", extra="audio", purpose="audio decoding")
    data, native_sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return resample(mono, int(native_sr), sr)
