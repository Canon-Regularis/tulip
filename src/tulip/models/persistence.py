"""Persistence for trained tulip models and pipelines.

A trained model (or full sklearn ``Pipeline``) is saved as a directory
artifact with two files:

* ``model.joblib`` — the estimator, serialised with joblib.
* ``metadata.json`` — a deterministic JSON sidecar recording the environment
  (tulip and Python versions), the model class, the fitted class labels when
  available, and arbitrary user metadata (resolved config, metrics, ...).

The sidecar is deterministic by construction — keys are sorted and no
timestamps are written — so saving the same model twice produces
byte-identical metadata, keeping artifacts diff-friendly and content-hashable
for experiment tracking.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.utils.logging import get_logger

logger = get_logger(__name__)

#: File name of the joblib-serialised estimator inside a model directory.
MODEL_FILENAME = "model.joblib"
#: File name of the JSON sidecar inside a model directory.
METADATA_FILENAME = "metadata.json"
#: Version of the on-disk artifact layout, recorded in the sidecar.
FORMAT_VERSION = 1

__all__ = [
    "FORMAT_VERSION",
    "METADATA_FILENAME",
    "MODEL_FILENAME",
    "load_model",
    "save_model",
]


def _tulip_version() -> str:
    """Return the installed tulip version (dev fallback handled by the package)."""
    import tulip

    return getattr(tulip, "__version__", "unknown")


def _json_default(value: Any) -> Any:
    """Coerce common non-JSON values (numpy scalars/arrays, ``Path``) to JSON.

    Anything else raises ``TypeError`` so :func:`save_model` can fail loudly
    instead of writing a lossy or non-deterministic sidecar.
    """
    if isinstance(value, Path):
        return str(value)
    to_list = getattr(value, "tolist", None)  # numpy scalars and arrays
    if callable(to_list):
        return to_list()
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def _classes_of(model: Any) -> list[Any] | None:
    """Return the fitted class labels of ``model`` as a JSON-friendly list.

    Returns ``None`` when the model is unfitted or exposes no ``classes_``
    (sklearn ``Pipeline`` raises from the ``classes_`` property before fit,
    hence the broad guard).
    """
    try:
        classes = getattr(model, "classes_", None)
    except Exception:
        return None
    if classes is None:
        return None
    to_list = getattr(classes, "tolist", None)
    return to_list() if callable(to_list) else list(classes)


def save_model(model: Any, path: Path | str, metadata: Mapping[str, Any] | None = None) -> Path:
    """Save a trained model (or pipeline) plus a deterministic JSON sidecar.

    Writes ``<path>/model.joblib`` and ``<path>/metadata.json``, creating the
    directory if needed and overwriting an existing artifact in place. The
    sidecar is serialised (and thus validated) before anything touches disk,
    so a bad ``metadata`` value never leaves a partial artifact behind.

    Args:
        model: A fitted (or unfitted) estimator/pipeline to serialise.
        path: Directory to write the artifact into.
        metadata: User metadata (e.g. resolved experiment config, metrics),
            stored under the sidecar's ``"metadata"`` key. Values must be
            JSON-serialisable; numpy scalars/arrays and ``Path`` are coerced.

    Returns:
        The artifact directory as a ``Path``.

    Raises:
        ConfigurationError: If ``metadata`` is not JSON-serialisable.
        DataError: If ``path`` exists and is not a directory.
    """
    target = Path(path)
    sidecar: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "tulip_version": _tulip_version(),
        "python_version": platform.python_version(),
        "classes": _classes_of(model),
        "metadata": dict(metadata or {}),
    }
    try:
        payload = json.dumps(
            sidecar, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"model metadata is not JSON-serialisable: {exc}") from exc

    if target.exists() and not target.is_dir():
        raise DataError(f"cannot save model: {target} exists and is not a directory")
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / MODEL_FILENAME)
    (target / METADATA_FILENAME).write_text(payload + "\n", encoding="utf-8", newline="\n")
    logger.info("saved %s to %s", sidecar["model_class"], target)
    return target


def load_model(path: Path | str) -> tuple[Any, dict[str, Any]]:
    """Load a model artifact written by :func:`save_model`.

    Args:
        path: The artifact directory containing ``model.joblib`` and
            ``metadata.json``.

    Returns:
        ``(model, metadata)`` where ``metadata`` is the parsed sidecar dict
        (user metadata lives under its ``"metadata"`` key).

    Raises:
        DataError: If the directory or either file is missing, or a file is
            corrupt/unparseable.
        MissingDependencyError: Propagated from unpickling when the model's
            library (e.g. xgboost) is not installed; not masked as corruption.
    """
    target = Path(path)
    if not target.is_dir():
        raise DataError(f"model directory not found: {target}")
    model_path = target / MODEL_FILENAME
    metadata_path = target / METADATA_FILENAME
    missing = [p.name for p in (model_path, metadata_path) if not p.is_file()]
    if missing:
        raise DataError(f"model artifact at {target} is incomplete: missing {', '.join(missing)}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DataError(f"corrupt metadata sidecar at {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise DataError(
            f"corrupt metadata sidecar at {metadata_path}: "
            f"expected a JSON object, got {type(metadata).__name__}"
        )

    try:
        model = joblib.load(model_path)
    except (ImportError, MemoryError):
        raise  # a missing library is an environment problem, not file corruption
    except Exception as exc:
        raise DataError(
            f"failed to load model from {model_path}: file is corrupt or incompatible ({exc})"
        ) from exc
    logger.info("loaded %s from %s", metadata.get("model_class", type(model).__name__), target)
    return model, metadata
