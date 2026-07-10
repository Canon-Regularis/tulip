"""Pooling of frame-level audio features into fixed-size vectors.

Frame-level extractors produce ``(n_frames, n_dims)`` matrices whose length
depends on the audio duration; classifiers need one fixed-width row per file.
:func:`pool_features` collapses the time axis with summary statistics
(mean+std by default, optionally median/min/max). Pooling is NaN-safe because
several extractors legitimately emit NaN frames (unvoiced pitch frames,
formant-less frames): NaNs are ignored per column, and columns with no valid
frames pool to ``0.0`` so downstream models never see NaN or infinity.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from tulip.core.exceptions import ConfigurationError, DataError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Statistics applied by default: mean+std concatenation.
DEFAULT_STATS: tuple[str, ...] = ("mean", "std")

_STAT_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
}

#: All supported pooling statistic names.
VALID_STATS: tuple[str, ...] = tuple(_STAT_FUNCTIONS)

__all__ = [
    "DEFAULT_STATS",
    "VALID_STATS",
    "pool_features",
    "pooled_feature_names",
    "validate_stats",
]


def validate_stats(stats: Sequence[str]) -> tuple[str, ...]:
    """Validate pooling statistic names, returning them as a tuple.

    Args:
        stats: Statistic names; each must be one of :data:`VALID_STATS`.

    Returns:
        The validated statistics, in the given order.

    Raises:
        ConfigurationError: If ``stats`` is empty or contains unknown names.
    """
    resolved = (stats,) if isinstance(stats, str) else tuple(stats)
    if not resolved:
        raise ConfigurationError("at least one pooling statistic is required")
    unknown = [stat for stat in resolved if stat not in _STAT_FUNCTIONS]
    if unknown:
        raise ConfigurationError(
            f"unknown pooling statistic(s) {unknown!r}; valid statistics: {VALID_STATS}"
        )
    return resolved


def pool_features(frames: np.ndarray, stats: Sequence[str] = DEFAULT_STATS) -> np.ndarray:
    """Pool an ``(n_frames, n_dims)`` matrix into a fixed vector of column statistics.

    The output concatenates one block per statistic (all dimensions for
    ``stats[0]``, then all dimensions for ``stats[1]``, ...), matching the
    order of :func:`pooled_feature_names`. Empty inputs (zero frames), NaN
    frames, and all-NaN columns pool to ``0.0`` rather than propagating NaN,
    so short or silent recordings still yield finite feature rows.

    Args:
        frames: Frame-level features, shape ``(n_frames, n_dims)``; a 1-D
            array is treated as a single-dimension feature track.
        stats: Statistics to compute per dimension (see :data:`VALID_STATS`).

    Returns:
        1-D float32 vector of length ``len(stats) * n_dims``, always finite.

    Raises:
        ConfigurationError: If ``stats`` is empty or contains unknown names.
        DataError: If ``frames`` has more than two dimensions.
    """
    resolved = validate_stats(stats)
    matrix = np.asarray(frames, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise DataError(f"frame matrix must be 1-D or 2-D, got shape {matrix.shape}")
    n_dims = matrix.shape[1]
    if matrix.shape[0] == 0 or n_dims == 0:
        return np.zeros(len(resolved) * n_dims, dtype=np.float32)
    with warnings.catch_warnings():
        # nan* functions warn on all-NaN columns; those pool to 0.0 below.
        warnings.simplefilter("ignore", RuntimeWarning)
        blocks = [_STAT_FUNCTIONS[stat](matrix, axis=0) for stat in resolved]
    pooled = np.concatenate(blocks)
    return np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pooled_feature_names(
    base_names: Sequence[str], stats: Sequence[str] = DEFAULT_STATS
) -> list[str]:
    """Return output feature names matching the layout of :func:`pool_features`.

    Args:
        base_names: Names of the per-frame dimensions (before pooling).
        stats: Statistics in the same order passed to :func:`pool_features`.

    Returns:
        Names like ``["mfcc_0_mean", ..., "mfcc_0_std", ...]``, one block per
        statistic.

    Raises:
        ConfigurationError: If ``stats`` is empty or contains unknown names.
    """
    resolved = validate_stats(stats)
    return [f"{name}_{stat}" for stat in resolved for name in base_names]
