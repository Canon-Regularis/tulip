"""Helpers shared by the two neural speech classifiers.

The fine-tuning wrapper (:mod:`tulip.models.neural_audio_finetune`) and the
frozen-embedding wrapper (:mod:`tulip.models.neural_audio_embedding`) both decode
audio the same way and both special-case Whisper's fixed-window feature
extractor. Those two helpers, and the shared target sample rate, live here so
neither wrapper has to import the other. No heavy dependency is imported at
module load; decoding goes through the canonical shared audio loader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.core.exceptions import DataError, TulipError
from tulip.features.audio.loading import DEFAULT_SAMPLE_RATE, load_audio

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np

__all__ = ["TARGET_SAMPLE_RATE", "is_whisper_extractor", "load_clipped_waveforms"]

#: Sample rate every model input is resampled to (Hz); one constant shared with
#: the audio feature extractors so the two subsystems cannot drift.
TARGET_SAMPLE_RATE = DEFAULT_SAMPLE_RATE


def is_whisper_extractor(feature_extractor: Any) -> bool:
    """Whether the extractor produces Whisper's fixed-window ``input_features``.

    Whisper's feature extractor pads/truncates every clip to a fixed 30 s
    log-mel window, so batches must not use dynamic ``padding=True``.
    Detection is by class name to avoid importing transformers here.

    Args:
        feature_extractor: A Hugging Face feature extractor instance.

    Returns:
        ``True`` for Whisper-style extractors.
    """
    return type(feature_extractor).__name__.lower().startswith("whisper")


def load_clipped_waveforms(
    paths: Sequence[str | Path], *, sample_rate: int, max_seconds: float
) -> list[np.ndarray]:
    """Decode audio files to mono waveforms, clipped to ``max_seconds``.

    Args:
        paths: Audio files to decode.
        sample_rate: Target sample rate in Hz.
        max_seconds: Maximum clip duration retained per file.

    Returns:
        One 1-D ``float32`` waveform per input path.

    Raises:
        DataError: if a file cannot be decoded or decodes to an empty waveform.
    """
    limit = max(1, round(max_seconds * sample_rate))
    waveforms: list[np.ndarray] = []
    for path in paths:
        try:
            waveform = load_audio(path, sample_rate=sample_rate)
        except TulipError:
            raise
        except Exception as exc:
            raise DataError(f"failed to decode audio file {path}: {exc}") from exc
        if waveform.size == 0:
            raise DataError(f"audio file {path} decoded to an empty waveform")
        waveforms.append(waveform[:limit])
    return waveforms
