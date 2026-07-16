"""Shared validation helpers for the viz builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Rendering backends shared by the chart and plot builders.
_BACKENDS = ("matplotlib", "plotly")

# Palette roles (validated single-hue set) shared by every chart builder:
# magnitude bars stay in one hue, with the winning class emphasised by a darker
# step, not a new hue. Kept in one place so the chart modules cannot drift.
_BAR_COLOR = "#86b6ef"
_WINNER_COLOR = "#256abf"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"


def _validate_choice(value: str, choices: Sequence[str], noun: str) -> str:
    """Normalise ``value`` and check it against ``choices``.

    Returns the stripped, lower-cased key. Raises
    :class:`~tulip.core.exceptions.ConfigurationError` naming ``noun`` and the
    original ``value`` when the key is not one of ``choices``.
    """
    key = value.strip().lower()
    if key not in choices:
        raise ConfigurationError(f"unknown {noun} {value!r}; expected one of {', '.join(choices)}")
    return key


def _validate_backend(backend: str) -> str:
    """Normalise and validate a chart/plot rendering backend name."""
    return _validate_choice(backend, _BACKENDS, "chart backend")
