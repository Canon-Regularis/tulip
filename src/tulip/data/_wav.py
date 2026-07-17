"""Deterministic int16 PCM WAV writer shared by the audio producers.

The synthetic-audio generator and the Hub audio cache both quantise a float
signal to mono int16 PCM and write it with the standard-library :mod:`wave`
module, byte-for-byte identically. This holds that one writer so the quantisation
and byte layout live in a single place; the synthetic-audio determinism guarantee
and the content-addressed Hub cache both depend on those bytes never drifting.

numpy is imported lazily inside the writer so importing this module (and the data
package that pulls it) stays light for the text-only pipelines.
"""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

__all__ = ["INT16_MAX", "INT16_MIN", "write_int16_pcm_wav"]

#: Full-scale int16 bounds; a float signal in ``[-1, 1]`` scales onto these.
INT16_MAX = 32_767
INT16_MIN = -32_768


def write_int16_pcm_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    """Quantise a 1-D float signal to int16 PCM (no dithering) and write a mono WAV.

    The explicit little-endian ``<i2`` dtype makes the bytes platform independent,
    which both the synthetic-audio determinism guarantee and the Hub audio cache
    rely on. The caller passes a 1-D signal; a multi-channel source must be
    downmixed to mono first.

    Args:
        path: Destination WAV file.
        signal: 1-D float signal, nominally in ``[-1, 1]``.
        sample_rate: Output sample rate in Hz.
    """
    import numpy as np

    quantised = np.clip(np.round(signal * INT16_MAX), INT16_MIN, INT16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(quantised.tobytes())
