"""Shared scaffolding for pooled frame-level audio feature extractors.

Concrete extractors implement :meth:`PooledFrameExtractor._frame_features`
(one ``(n_frames, n_dims)`` matrix per decoded signal) and inherit uniform
loading, pooling, and ``get_feature_names_out`` behaviour. Extractors are
stateless sklearn transformers: ``fit`` is a no-op and ``transform`` maps a
sequence of audio file paths to one pooled float32 row per file.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from tulip.features.audio.loading import DEFAULT_SAMPLE_RATE, load_audio
from tulip.features.audio.pooling import (
    DEFAULT_STATS,
    pool_features,
    pooled_feature_names,
    validate_stats,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["PooledFrameExtractor"]


class PooledFrameExtractor(TransformerMixin, BaseEstimator, abc.ABC):
    """Base class turning frame-level audio features into fixed-size rows.

    Subclasses only compute the per-frame feature matrix; decoding (via
    :func:`~tulip.features.audio.loading.load_audio`), NaN-safe pooling, and
    feature naming are shared here so every audio feature behaves identically
    inside :class:`sklearn.pipeline.FeatureUnion`.

    Args:
        sample_rate: Sample rate every file is resampled to before extraction.
        stats: Pooling statistics (see
            :data:`tulip.features.audio.pooling.VALID_STATS`).
    """

    def __init__(
        self, sample_rate: int = DEFAULT_SAMPLE_RATE, stats: Sequence[str] = DEFAULT_STATS
    ) -> None:
        self.sample_rate = sample_rate
        self.stats = stats

    def fit(self, X: Sequence[str | Path], y: Any = None) -> PooledFrameExtractor:
        """Validate parameters and return ``self`` (extractors are stateless).

        Validating ``stats`` here rather than only in ``transform`` surfaces a
        misconfiguration when the pipeline is *fitted*, not after the whole
        pipeline has already been trained.

        Raises:
            ConfigurationError: If ``stats`` contains an unknown statistic.
        """
        del X, y
        validate_stats(self.stats)
        return self

    def transform(self, X: Sequence[str | Path]) -> np.ndarray:
        """Extract one pooled feature row per audio file.

        Args:
            X: Sequence of audio file paths.

        Returns:
            Dense float32 array of shape ``(len(X), len(stats) * n_dims)``.

        Raises:
            DataError: If a file is missing or cannot be decoded.
            MissingDependencyError: If a required optional dependency is not
                installed.
        """
        stats = validate_stats(self.stats)
        n_base = len(self._base_feature_names())
        rows: list[np.ndarray] = []
        for path in X:
            signal = load_audio(path, sample_rate=self.sample_rate)
            if signal.size == 0:
                frames = np.zeros((0, n_base), dtype=np.float64)
            else:
                frames = self._frame_features(signal, self.sample_rate)
            rows.append(pool_features(frames, stats=stats))
        if not rows:
            return np.zeros((0, len(stats) * n_base), dtype=np.float32)
        return np.vstack(rows)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output column names (sklearn convention)."""
        del input_features
        names = pooled_feature_names(self._base_feature_names(), validate_stats(self.stats))
        return np.asarray(names, dtype=object)

    @abc.abstractmethod
    def _frame_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute the ``(n_frames, n_dims)`` frame-level feature matrix.

        Implementations may return NaN entries for undefined frames (e.g.
        unvoiced pitch frames); pooling handles them.
        """

    @abc.abstractmethod
    def _base_feature_names(self) -> list[str]:
        """Names of the per-frame dimensions, before pooling."""
