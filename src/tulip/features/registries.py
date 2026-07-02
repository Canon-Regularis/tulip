"""Registries of feature extractors, keyed by canonical name."""

from __future__ import annotations

from typing import Any

from tulip.core.registry import Registry

#: Text feature extractors (operate on sequences of strings).
TEXT_FEATURES: Registry[Any] = Registry("text feature")

#: Audio feature extractors (operate on sequences of audio file paths).
AUDIO_FEATURES: Registry[Any] = Registry("audio feature")
