"""Spectral audio features: MFCC, log-mel, RMS energy, ZCR, centroid, chroma.

All extractors are pooled frame-level features registered in
:data:`tulip.features.AUDIO_FEATURES` under ``mfcc``, ``mel_spectrogram``,
``energy``, ``zero_crossing_rate``, ``spectral_centroid``, and ``chroma``.
librosa is imported lazily inside the frame computations (``audio`` extra).

Framing defaults (``n_fft=1024``, ``hop_length=256``) correspond to 64 ms
windows with a 16 ms hop at 16 kHz — finer-grained than librosa's music
defaults, which suits short dialect speech recordings.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType

import numpy as np

from tulip.features.audio._base import PooledFrameExtractor
from tulip.features.audio.loading import DEFAULT_SAMPLE_RATE
from tulip.features.audio.pooling import DEFAULT_STATS
from tulip.features.registries import AUDIO_FEATURES
from tulip.utils import optional
from tulip.utils.logging import get_logger

logger = get_logger(__name__)

#: STFT window size in samples (64 ms at 16 kHz).
DEFAULT_N_FFT = 1024
#: STFT hop size in samples (16 ms at 16 kHz).
DEFAULT_HOP_LENGTH = 256

__all__ = [
    "ChromaExtractor",
    "MelSpectrogramExtractor",
    "MfccExtractor",
    "RmsEnergyExtractor",
    "SpectralCentroidExtractor",
    "ZeroCrossingRateExtractor",
]


def _librosa() -> ModuleType:
    """Lazily import librosa (provided by the ``audio`` extra)."""
    return optional.optional_import(
        "librosa", extra="audio", purpose="spectral audio feature extraction"
    )


@AUDIO_FEATURES.register("mfcc")
class MfccExtractor(PooledFrameExtractor):
    """Pooled MFCCs, optionally augmented with delta and delta-delta tracks.

    Args:
        sample_rate: Target sample rate.
        n_mfcc: Number of cepstral coefficients per frame.
        add_deltas: Append first- and second-order deltas (triples the width).
        n_fft: STFT window size in samples.
        hop_length: STFT hop size in samples.
        stats: Pooling statistics.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_mfcc: int = 13,
        add_deltas: bool = False,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        stats: Sequence[str] = DEFAULT_STATS,
    ) -> None:
        super().__init__(sample_rate=sample_rate, stats=stats)
        self.n_mfcc = n_mfcc
        self.add_deltas = add_deltas
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _frame_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        librosa = _librosa()
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        if self.add_deltas:
            # mode="nearest" keeps deltas defined for clips shorter than the
            # default 9-frame regression window.
            delta = librosa.feature.delta(mfcc, mode="nearest")
            delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
            mfcc = np.vstack([mfcc, delta, delta2])
        return mfcc.T

    def _base_feature_names(self) -> list[str]:
        names = [f"mfcc_{i}" for i in range(self.n_mfcc)]
        if self.add_deltas:
            names += [f"mfcc_delta_{i}" for i in range(self.n_mfcc)]
            names += [f"mfcc_delta2_{i}" for i in range(self.n_mfcc)]
        return names


@AUDIO_FEATURES.register("mel_spectrogram")
class MelSpectrogramExtractor(PooledFrameExtractor):
    """Pooled log-mel spectrogram (power spectrogram in dB on a mel scale).

    Args:
        sample_rate: Target sample rate.
        n_mels: Number of mel bands.
        n_fft: STFT window size in samples.
        hop_length: STFT hop size in samples.
        fmin: Lowest mel-filter frequency in Hz.
        fmax: Highest mel-filter frequency in Hz (``None`` = Nyquist).
        stats: Pooling statistics.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_mels: int = 64,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        fmin: float = 0.0,
        fmax: float | None = None,
        stats: Sequence[str] = DEFAULT_STATS,
    ) -> None:
        super().__init__(sample_rate=sample_rate, stats=stats)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def _frame_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        librosa = _librosa()
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        # Absolute dB reference (ref=1.0) keeps loudness information comparable
        # across files instead of normalising each file to its own peak.
        return librosa.power_to_db(mel, ref=1.0).T

    def _base_feature_names(self) -> list[str]:
        return [f"mel_{i}" for i in range(self.n_mels)]


@AUDIO_FEATURES.register("energy")
class RmsEnergyExtractor(PooledFrameExtractor):
    """Pooled frame-wise RMS energy.

    Args:
        sample_rate: Target sample rate.
        frame_length: Analysis window size in samples.
        hop_length: Hop size in samples.
        stats: Pooling statistics.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_length: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        stats: Sequence[str] = DEFAULT_STATS,
    ) -> None:
        super().__init__(sample_rate=sample_rate, stats=stats)
        self.frame_length = frame_length
        self.hop_length = hop_length

    def _frame_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        librosa = _librosa()
        rms = librosa.feature.rms(
            y=signal, frame_length=self.frame_length, hop_length=self.hop_length
        )
        return rms.T

    def _base_feature_names(self) -> list[str]:
        return ["rms"]


@AUDIO_FEATURES.register("zero_crossing_rate")
class ZeroCrossingRateExtractor(PooledFrameExtractor):
    """Pooled frame-wise zero-crossing rate (crossings per sample).

    Args:
        sample_rate: Target sample rate.
        frame_length: Analysis window size in samples.
        hop_length: Hop size in samples.
        stats: Pooling statistics.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_length: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        stats: Sequence[str] = DEFAULT_STATS,
    ) -> None:
        super().__init__(sample_rate=sample_rate, stats=stats)
        self.frame_length = frame_length
        self.hop_length = hop_length

    def _frame_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        librosa = _librosa()
        zcr = librosa.feature.zero_crossing_rate(
            y=signal, frame_length=self.frame_length, hop_length=self.hop_length
        )
        return zcr.T

    def _base_feature_names(self) -> list[str]:
        return ["zcr"]


@AUDIO_FEATURES.register("spectral_centroid")
class SpectralCentroidExtractor(PooledFrameExtractor):
    """Pooled spectral centroid (the spectrum's centre of mass, in Hz).

    Args:
        sample_rate: Target sample rate.
        n_fft: STFT window size in samples.
        hop_length: STFT hop size in samples.
        stats: Pooling statistics.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        stats: Sequence[str] = DEFAULT_STATS,
    ) -> None:
        super().__init__(sample_rate=sample_rate, stats=stats)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _frame_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        librosa = _librosa()
        centroid = librosa.feature.spectral_centroid(
            y=signal, sr=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return centroid.T

    def _base_feature_names(self) -> list[str]:
        return ["spectral_centroid"]


@AUDIO_FEATURES.register("chroma")
class ChromaExtractor(PooledFrameExtractor):
    """Pooled chromagram (energy per pitch class).

    Args:
        sample_rate: Target sample rate.
        n_chroma: Number of pitch-class bins.
        n_fft: STFT window size in samples.
        hop_length: STFT hop size in samples.
        stats: Pooling statistics.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_chroma: int = 12,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        stats: Sequence[str] = DEFAULT_STATS,
    ) -> None:
        super().__init__(sample_rate=sample_rate, stats=stats)
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _frame_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        librosa = _librosa()
        chroma = librosa.feature.chroma_stft(
            y=signal,
            sr=sample_rate,
            n_chroma=self.n_chroma,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return chroma.T

    def _base_feature_names(self) -> list[str]:
        return [f"chroma_{i}" for i in range(self.n_chroma)]
