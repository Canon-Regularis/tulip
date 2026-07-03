"""The DATASETS registry: canonical names -> :class:`DatasetLoader` classes.

Loader modules under :mod:`tulip.data.loaders` register themselves here at
import time; importing :mod:`tulip.data` triggers those imports so the
registry is fully populated as soon as the package is used.
"""

from __future__ import annotations

from tulip.core.interfaces import DatasetLoader
from tulip.core.registry import Registry

#: Dataset loaders, keyed by canonical corpus name (see docs/architecture.md).
DATASETS: Registry[type[DatasetLoader]] = Registry("dataset")

__all__ = ["DATASETS"]
