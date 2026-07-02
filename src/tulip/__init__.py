"""tulip: Polish Dialect Typology and Regional Speech Classification Analysis System.

A modular toolkit for detecting Polish dialects from written text, transcribed
speech, and raw audio, with support for classical ML baselines, transformer
models, explainability, and map-based visualisation.

Heavy subsystems are imported lazily; ``import tulip`` stays cheap.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("tulip")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.0.0.dev0"

_LAZY_EXPORTS = {
    "Sample": "tulip.core.types",
    "Prediction": "tulip.core.types",
    "Explanation": "tulip.core.types",
    "DialectLabels": "tulip.core.types",
    "DialectFamily": "tulip.labels.taxonomy",
    "RegionalDialect": "tulip.labels.taxonomy",
    "ExperimentConfig": "tulip.config.schemas",
    "DialectClassifier": "tulip.pipeline.classifier",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Resolve top-level exports lazily to keep ``import tulip`` lightweight."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
