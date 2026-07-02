"""Registry of classifier models, keyed by canonical name."""

from __future__ import annotations

from typing import Any

from tulip.core.registry import Registry

#: All classifier models (classical, transformer text, and speech).
MODELS: Registry[Any] = Registry("model")
