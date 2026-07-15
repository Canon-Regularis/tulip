"""Shared validation helpers for the viz builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Rendering backends shared by the chart and plot builders.
_BACKENDS = ("matplotlib", "plotly")


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
