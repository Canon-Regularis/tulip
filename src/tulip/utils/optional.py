"""Controlled access to optional heavy dependencies.

tulip keeps its core installation light; torch, librosa, shap, folium, etc.
are optional extras. Modules must import such dependencies lazily -- inside
functions or methods -- through :func:`optional_import`, which converts an
``ImportError`` into an actionable :class:`MissingDependencyError` naming the
pip extra to install.
"""

from __future__ import annotations

import importlib
import importlib.util
from functools import cache
from types import ModuleType

from tulip.core.exceptions import MissingDependencyError


def optional_import(
    module: str, *, extra: str | None = None, purpose: str | None = None
) -> ModuleType:
    """Import ``module``, raising a helpful error if it is not installed.

    Args:
        module: Dotted module path to import (e.g. ``"librosa"``).
        extra: The pip extra that provides it (e.g. ``"audio"``), used in the
            error message's install hint.
        purpose: Short description of the feature needing it, for the error.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: if the module cannot be imported.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(module, extra=extra, purpose=purpose) from exc


@cache
def is_available(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False
