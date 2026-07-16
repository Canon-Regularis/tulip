"""The model registry's value model and pure entry operations.

The data types and stateless logic behind the registry: the lifecycle
:class:`Stage`, the immutable :class:`RegistryEntry`, the on-disk index shape,
the content digest, and the pure helpers that find, restage, and select entries.
:mod:`tulip.deploy.registry_store` composes these into the stateful
:class:`~tulip.deploy.registry_store.ModelRegistry` that owns the filesystem
I/O. Keeping the value model and its pure operations here makes them testable
without touching disk, and lets a consumer depend on the entry shape without
pulling the whole store.
"""

from __future__ import annotations

import enum
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import DataError
from tulip.models.persistence import METADATA_FILENAME, MODEL_FILENAME

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["RegistryEntry", "Stage", "artifact_digest"]

#: Schema version of the on-disk index; bump on a breaking change.
_INDEX_SCHEMA_VERSION = 1
#: The two files that make up a persisted artifact, hashed in this fixed order.
_ARTIFACT_FILES = (MODEL_FILENAME, METADATA_FILENAME)


class Stage(str, enum.Enum):
    """Lifecycle stage of a registered model version."""

    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class RegistryEntry(BaseModel):
    """One immutable registered ``(name, version)`` model version.

    ``digest`` is the content address; ``model_class``/``classes``/``target``/
    ``task`` are read from the persisted sidecar so the entry is self-describing
    without loading the model.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    digest: str
    stage: Stage
    model_class: str
    tulip_version: str
    target: str | None = None
    task: str | None = None
    classes: tuple[str, ...] = ()
    metrics: dict[str, float] | None = None


def artifact_digest(model_dir: Path | str) -> str:
    """Return the SHA-256 content digest of a persisted model artifact.

    Hashes ``model.joblib`` and ``metadata.json`` in a fixed order, each
    length-prefixed so no rearrangement of bytes across the two files can collide.

    Raises:
        DataError: if either artifact file is missing or unreadable.
    """
    directory = Path(model_dir)
    hasher = hashlib.sha256()
    for name in _ARTIFACT_FILES:
        try:
            data = (directory / name).read_bytes()
        except OSError as exc:
            raise DataError(f"cannot digest artifact at {directory}: {exc}") from exc
        hasher.update(f"{name}:{len(data)}\0".encode())
        hasher.update(data)
    return hasher.hexdigest()


class _Index(BaseModel):
    """On-disk registry index (entries in registration order + promotion stacks)."""

    schema_version: int = _INDEX_SCHEMA_VERSION
    entries: list[RegistryEntry] = Field(default_factory=list)
    history: dict[str, list[str]] = Field(default_factory=dict)


def _restage(
    entry: RegistryEntry, name: str, *, current: str | None, previous: str
) -> RegistryEntry:
    """Archive ``name``'s current production and set ``previous`` to production."""
    if entry.name != name:
        return entry
    if current is not None and entry.version == current:
        return entry.model_copy(update={"stage": Stage.ARCHIVED})
    if entry.version == previous:
        return entry.model_copy(update={"stage": Stage.PRODUCTION})
    return entry


def _find(entries: Sequence[RegistryEntry], name: str, version: str) -> RegistryEntry | None:
    """The entry for ``(name, version)``, or ``None``."""
    return next(
        (entry for entry in entries if entry.name == name and entry.version == version), None
    )


def _meta(sidecar: dict[str, Any], key: str) -> str | None:
    """A value from the sidecar's user ``metadata`` block, coerced to ``str``."""
    metadata = sidecar.get("metadata")
    value = metadata.get(key) if isinstance(metadata, dict) else None
    return str(value) if value is not None else None


def _by_stage(entries: Sequence[RegistryEntry], stage: Stage, reference: str) -> RegistryEntry:
    """The most recently added entry of ``name`` at ``stage`` (production is unique)."""
    matches = [entry for entry in entries if entry.stage is stage]
    if not matches:
        raise DataError(f"{reference}: no version is in {stage.value}")
    return matches[-1]
