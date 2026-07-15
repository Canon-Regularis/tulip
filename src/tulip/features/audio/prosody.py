"""Prosodic audio features: pitch (F0) statistics and formant frequencies.

Registers ``pitch`` and ``formants`` in :data:`tulip.features.AUDIO_FEATURES`.

``pitch`` summarises the probabilistic-YIN F0 track (librosa, ``audio``
extra) over voiced frames only, so unvoiced/silent stretches never poison the
statistics with NaN.

``formants`` prefers Praat's Burg algorithm (praat-parselmouth) and falls
back to a classic autocorrelation-LPC estimate (:func:`lpc_formant_frames`,
pure numpy/scipy) when parselmouth is unavailable, so the feature works from
the ``audio`` extra alone — and, given pre-decoded signals, with no optional
dependency at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import solve_toeplitz
from sklearn.base import BaseEstimator, TransformerMixin

from tulip.core.exceptions import ConfigurationError
from tulip.features.audio._base import PooledFrameExtractor
from tulip.features.audio.loading import DEFAULT_SAMPLE_RATE, load_audio
from tulip.features.audio.pooling import DEFAULT_STATS
from tulip.features.registries import AUDIO_FEATURES
from tulip.utils import optional
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = get_logger(__name__)

__all__ = ["FormantExtractor", "PitchExtractor", "lpc_formant_frames"]

_PITCH_FEATURE_NAMES = ("f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "voiced_ratio")


@AUDIO_FEATURES.register("pitch", metadata={"extra": "audio"})
class PitchExtractor(TransformerMixin, BaseEstimator):
    """Voiced-frame F0 statistics from librosa's probabilistic YIN tracker.

    Each file maps to six values: mean, standard deviation, minimum, maximum,
    and range of F0 over voiced frames, plus the voiced-frame ratio. Files
    with no voiced frames yield an all-zero row rather than NaNs.

    Args:
        sample_rate: Target sample rate.
        fmin: Lowest F0 candidate in Hz.
        fmax: Highest F0 candidate in Hz.
        frame_length: pyin analysis window in samples.
        hop_length: pyin hop size in samples.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        fmin: float = 65.0,
        fmax: float = 500.0,
        frame_length: int = 2048,
        hop_length: int = 256,
    ) -> None:
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.hop_length = hop_length

    def fit(self, X: Sequence[str | Path], y: Any = None) -> PitchExtractor:
        """No-op fit (the extractor is stateless); returns ``self``."""
        del X, y
        return self

    def transform(self, X: Sequence[str | Path]) -> np.ndarray:
        """Return one row of F0 statistics per audio file.

        Args:
            X: Sequence of audio file paths.

        Returns:
            Float32 array of shape ``(len(X), 6)``, always finite.
        """
        rows = [self._pitch_stats(load_audio(path, sample_rate=self.sample_rate)) for path in X]
        if not rows:
            return np.zeros((0, len(_PITCH_FEATURE_NAMES)), dtype=np.float32)
        return np.vstack(rows)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output column names (sklearn convention)."""
        del input_features
        return np.asarray(_PITCH_FEATURE_NAMES, dtype=object)

    def _pitch_stats(self, signal: np.ndarray) -> np.ndarray:
        librosa = optional.optional_import(
            "librosa", extra="audio", purpose="F0 tracking with probabilistic YIN"
        )
        if signal.size == 0:
            return np.zeros(len(_PITCH_FEATURE_NAMES), dtype=np.float32)
        if signal.size < self.frame_length:
            signal = np.pad(signal, (0, self.frame_length - signal.size))
        f0, _voiced_flag, _voiced_prob = librosa.pyin(
            y=signal,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        voiced = f0[np.isfinite(f0)]
        if voiced.size == 0:
            return np.zeros(len(_PITCH_FEATURE_NAMES), dtype=np.float32)
        voiced_ratio = voiced.size / f0.size
        stats = np.array(
            [
                np.mean(voiced),
                np.std(voiced),
                np.min(voiced),
                np.max(voiced),
                np.max(voiced) - np.min(voiced),
                voiced_ratio,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


def lpc_formant_frames(
    signal: np.ndarray,
    sample_rate: int,
    *,
    n_formants: int = 3,
    frame_length: int = 400,
    hop_length: int = 160,
    lpc_order: int | None = None,
    pre_emphasis: float = 0.97,
    min_formant_hz: float = 90.0,
    max_formant_hz: float = 5500.0,
    max_bandwidth_hz: float = 400.0,
    energy_threshold: float = 0.05,
) -> np.ndarray:
    """Estimate per-frame formants with autocorrelation LPC (numpy/scipy only).

    This is the documented fallback used when praat-parselmouth is not
    installed. Classic source-filter formant estimation:

    1. Pre-emphasise (``y[t] - a * y[t-1]``) to flatten the glottal spectral
       tilt so higher formants influence the LP fit.
    2. Slice into Hamming-windowed frames (defaults: 25 ms / 10 ms at 16 kHz).
    3. Fit an all-pole model per frame with the autocorrelation method,
       solving the Toeplitz normal equations via Levinson-Durbin
       (``scipy.linalg.solve_toeplitz``). Order defaults to the ``2 + sample_rate/1000``
       rule of thumb.
    4. Root the LP polynomial; each positive-frequency root ``z`` is a
       resonance candidate at ``F = angle(z) * sample_rate / (2*pi)`` with bandwidth
       ``B = -(sample_rate / pi) * ln|z|``.
    5. Keep sharp resonances (``B < max_bandwidth_hz``) inside
       ``(min_formant_hz, max_formant_hz)``; the lowest ``n_formants`` in
       ascending frequency are F1..Fn.

    Frames below the energy gate (relative to the loudest frame — silence or
    weak unvoiced speech, where LPC roots are meaningless) and missing
    formants are returned as NaN; downstream pooling is NaN-safe.

    Args:
        signal: 1-D mono audio signal.
        sample_rate: Sample rate of ``signal`` in Hz.
        n_formants: Number of formants to keep per frame (F1..Fn).
        frame_length: Analysis window in samples.
        hop_length: Hop size in samples.
        lpc_order: All-pole model order; ``None`` uses ``2 + sample_rate // 1000``.
        pre_emphasis: Pre-emphasis coefficient in ``[0, 1)``.
        min_formant_hz: Reject candidates at or below this frequency.
        max_formant_hz: Reject candidates at or above this frequency.
        max_bandwidth_hz: Reject candidates broader than this bandwidth.
        energy_threshold: Frames with RMS below this fraction of the loudest
            frame's RMS are skipped (NaN row).

    Returns:
        Float array of shape ``(n_frames, n_formants)`` with NaN for
        undefined entries; ``(0, n_formants)`` when no frame fits.

    Raises:
        ConfigurationError: If the frame geometry or model order is invalid.
    """
    if frame_length <= 0 or hop_length <= 0:
        raise ConfigurationError(
            f"frame_length and hop_length must be positive, got {frame_length}, {hop_length}"
        )
    if n_formants <= 0:
        raise ConfigurationError(f"n_formants must be positive, got {n_formants}")
    order = int(lpc_order) if lpc_order is not None else 2 + sample_rate // 1000
    if order < 2:
        raise ConfigurationError(f"lpc_order must be at least 2, got {order}")

    data = np.asarray(signal, dtype=np.float64).ravel()
    if data.size == 0:
        return np.zeros((0, n_formants), dtype=np.float64)
    if data.size < frame_length:
        data = np.pad(data, (0, frame_length - data.size))

    emphasized = np.empty_like(data)
    emphasized[0] = data[0]
    emphasized[1:] = data[1:] - pre_emphasis * data[:-1]

    window = np.hamming(frame_length)
    n_frames = 1 + (data.size - frame_length) // hop_length
    starts = [i * hop_length for i in range(n_frames)]

    frame_rms = np.array([np.sqrt(np.mean(data[s : s + frame_length] ** 2)) for s in starts])
    peak_rms = float(frame_rms.max(initial=0.0))
    nan_row = np.full(n_formants, np.nan)
    if peak_rms <= 0.0:
        return np.tile(nan_row, (n_frames, 1))
    gate = energy_threshold * peak_rms

    rows: list[np.ndarray] = []
    for start, rms in zip(starts, frame_rms, strict=True):
        if rms < gate:
            rows.append(nan_row)
            continue
        frame = emphasized[start : start + frame_length] * window
        candidates = _lpc_resonances(frame, sample_rate, order)
        if candidates.size == 0:
            rows.append(nan_row)
            continue
        keep = candidates[
            (candidates[:, 0] > min_formant_hz)
            & (candidates[:, 0] < max_formant_hz)
            & (candidates[:, 1] < max_bandwidth_hz)
        ][:, 0]
        keep = np.sort(keep)[:n_formants]
        row = nan_row.copy()
        row[: keep.size] = keep
        rows.append(row)
    return np.vstack(rows)


def _lpc_resonances(frame: np.ndarray, sample_rate: int, order: int) -> np.ndarray:
    """Return ``(frequency_hz, bandwidth_hz)`` pairs of one frame's LP roots."""
    autocorr = np.correlate(frame, frame, mode="full")[frame.size - 1 : frame.size + order]
    if autocorr[0] <= 1e-12:
        return np.zeros((0, 2))
    # Tiny white-noise regularisation keeps the Toeplitz system well-posed on
    # nearly periodic frames.
    autocorr = autocorr.copy()
    autocorr[0] *= 1.0 + 1e-9
    try:
        lp_coeffs = solve_toeplitz((autocorr[:-1], autocorr[:-1]), -autocorr[1:])
    except (np.linalg.LinAlgError, ValueError):
        return np.zeros((0, 2))
    roots = np.roots(np.concatenate(([1.0], lp_coeffs)))
    roots = roots[np.imag(roots) > 0.0]
    if roots.size == 0:
        return np.zeros((0, 2))
    freqs = np.angle(roots) * sample_rate / (2.0 * np.pi)
    bandwidths = -(sample_rate / np.pi) * np.log(np.clip(np.abs(roots), 1e-12, None))
    return np.column_stack([freqs, bandwidths])


@AUDIO_FEATURES.register("formants", metadata={"extra": "audio"})
class FormantExtractor(PooledFrameExtractor):
    """Pooled F1..Fn formant-frequency tracks.

    Uses Praat's Burg algorithm through praat-parselmouth when available;
    otherwise the pure numpy/scipy LPC fallback (:func:`lpc_formant_frames`).
    Frames where a formant is undefined are NaN and ignored by pooling.

    Args:
        sample_rate: Target sample rate.
        n_formants: Number of formants per frame (F1..Fn).
        method: ``"auto"`` (praat if importable, else LPC), ``"praat"``, or
            ``"lpc"``.
        frame_length: Analysis window in samples (25 ms at 16 kHz default).
        hop_length: Hop size in samples (10 ms at 16 kHz default).
        lpc_order: LPC model order for the fallback; ``None`` = rule of thumb.
        max_formant_hz: Upper bound of the formant search range.
        stats: Pooling statistics.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_formants: int = 3,
        method: str = "auto",
        frame_length: int = 400,
        hop_length: int = 160,
        lpc_order: int | None = None,
        max_formant_hz: float = 5500.0,
        stats: Sequence[str] = DEFAULT_STATS,
    ) -> None:
        super().__init__(sample_rate=sample_rate, stats=stats)
        self.n_formants = n_formants
        self.method = method
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.lpc_order = lpc_order
        self.max_formant_hz = max_formant_hz

    def _frame_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        method = self._resolve_method()
        if method == "praat":
            return self._praat_formant_frames(signal, sample_rate)
        return lpc_formant_frames(
            signal,
            sample_rate,
            n_formants=self.n_formants,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            lpc_order=self.lpc_order,
            max_formant_hz=self.max_formant_hz,
        )

    def _base_feature_names(self) -> list[str]:
        return [f"f{i + 1}" for i in range(self.n_formants)]

    def _resolve_method(self) -> str:
        if self.method == "auto":
            if optional.is_available("parselmouth"):
                return "praat"
            logger.debug("parselmouth unavailable; using LPC formant fallback")
            return "lpc"
        if self.method in ("praat", "lpc"):
            return self.method
        raise ConfigurationError(
            f"unknown formant method {self.method!r}; expected 'auto', 'praat', or 'lpc'"
        )

    def _praat_formant_frames(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        parselmouth = optional.optional_import(
            "parselmouth", extra="audio", purpose="Praat (Burg) formant estimation"
        )
        sound = parselmouth.Sound(signal.astype(np.float64), sampling_frequency=sample_rate)
        formant = sound.to_formant_burg(
            time_step=self.hop_length / sample_rate,
            max_number_of_formants=max(5.0, float(self.n_formants)),
            maximum_formant=self.max_formant_hz,
            window_length=self.frame_length / sample_rate,
        )
        times = np.asarray(formant.ts(), dtype=np.float64)
        if times.size == 0:
            return np.zeros((0, self.n_formants), dtype=np.float64)
        rows = [
            [formant.get_value_at_time(index + 1, time) for index in range(self.n_formants)]
            for time in times
        ]
        return np.asarray(rows, dtype=np.float64)
