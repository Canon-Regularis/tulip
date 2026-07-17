"""Content-addressed model registry with a staging -> production promotion flow.

A trained model is a directory artifact (``model.joblib`` + ``metadata.json``,
see :mod:`tulip.models.persistence`) but nothing above it records *identity*: no
version, no integrity digest, and no notion of which artifact is the one in
production. This module adds that thin layer, and only that layer, reusing the
persistence format verbatim rather than re-implementing it:

* **Content addressing.** Every artifact is stored under
  ``<root>/artifacts/<sha256>/`` keyed by a SHA-256 over its bytes, so identical
  models deduplicate and bit-rot is caught as a digest mismatch rather than an
  opaque unpickle error.
* **Versioned entries.** Each ``(name, version)`` is an immutable
  :class:`RegistryEntry` recording the digest, the model class, and the labels,
  read straight from the persisted sidecar.
* **Promotion & rollback.** ``promote`` moves a version to ``production`` (the
  previous production is archived); ``rollback`` restores it in one call, using a
  per-name promotion stack so "the previous production" is unambiguous.
* **Resolution.** ``resolve("name@production")`` (or ``name@<version>``, or bare
  ``name`` for production) returns the entry, so the serving layer can bind to a
  moving target and report ``X-Model-Version`` / ``X-Model-Digest``.

The index (``<root>/registry.json``) is written with the shared deterministic
JSON writer (sorted keys, no timestamps) so the same operation sequence
reproduces byte-identical bytes, like the leaderboard provenance.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from tulip._jsonio import read_json_object
from tulip._serialize import write_sorted_json
from tulip.core.exceptions import ConfigurationError, DataError
from tulip.deploy._registry_model import (
    _ARTIFACT_FILES,
    RegistryEntry,
    Stage,
    _by_stage,
    _find,
    _Index,
    _meta,
    _restage,
    artifact_digest,
)
from tulip.models.persistence import METADATA_FILENAME
from tulip.utils.logging import get_logger

__all__ = [
    "REGISTRY_INDEX_NAME",
    "ModelRegistry",
    "RegistryEntry",
    "Stage",
    "artifact_digest",
]

logger = get_logger(__name__)

#: Index file name written under the registry root.
REGISTRY_INDEX_NAME = "registry.json"
#: Sub-directory holding the content-addressed artifacts.
_ARTIFACTS_DIR = "artifacts"


class ModelRegistry:
    """A content-addressed store of versioned models with a promotion flow.

    Args:
        root: Directory holding the index and the content-addressed artifacts;
            created on first write.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # ---------------------------------------------------------------- public

    def add(
        self,
        model_dir: Path | str,
        *,
        name: str,
        version: str,
        stage: Stage = Stage.STAGING,
        metrics: dict[str, float] | None = None,
    ) -> RegistryEntry:
        """Register a persisted model artifact under ``(name, version)``.

        Copies the artifact into content-addressed storage (deduplicating on
        digest) and appends an entry. Re-adding the identical artifact under the
        same ``(name, version)`` is idempotent.

        Raises:
            DataError: if ``model_dir`` is not a valid artifact.
            ConfigurationError: if ``(name, version)`` already exists with a
                *different* digest.
        """
        digest = artifact_digest(model_dir)
        sidecar = self._read_sidecar(model_dir)
        index = self._load_index()
        existing = _find(index.entries, name, version)
        if existing is not None:
            if existing.digest != digest:
                raise ConfigurationError(
                    f"{name}@{version} already exists with digest {existing.digest[:12]}; "
                    f"refusing to overwrite with {digest[:12]}"
                )
            return existing

        self._store_artifact(model_dir, digest)
        entry = RegistryEntry(
            name=name,
            version=version,
            digest=digest,
            stage=stage,
            model_class=str(sidecar.get("model_class", "unknown")),
            tulip_version=str(sidecar.get("tulip_version", "unknown")),
            target=_meta(sidecar, "target"),
            task=_meta(sidecar, "task"),
            classes=tuple(str(label) for label in sidecar.get("classes") or ()),
            metrics=metrics,
        )
        index.entries.append(entry)
        if stage is Stage.PRODUCTION:
            index = self._apply_promotion(index, name, version)
        self._save_index(index)
        logger.info("registered %s@%s (%s) digest=%s", name, version, stage.value, digest[:12])
        return _find(index.entries, name, version)  # type: ignore[return-value]  # just added

    def promote(self, name: str, version: str, *, stage: Stage = Stage.PRODUCTION) -> RegistryEntry:
        """Move ``(name, version)`` to ``stage``.

        Promoting to ``production`` archives the current production version of
        ``name`` and records the move on the promotion stack (so it can be rolled
        back). Promoting to ``staging``/``archived`` just relabels the entry.

        Raises:
            DataError: if ``(name, version)`` is not registered.
        """
        index = self._load_index()
        if _find(index.entries, name, version) is None:
            raise DataError(f"{name}@{version} is not registered")
        if stage is Stage.PRODUCTION:
            index = self._apply_promotion(index, name, version)
        else:
            index.entries = [
                entry.model_copy(update={"stage": stage})
                if entry.name == name and entry.version == version
                else entry
                for entry in index.entries
            ]
        self._save_index(index)
        logger.info("promoted %s@%s to %s", name, version, stage.value)
        return _find(index.entries, name, version)  # type: ignore[return-value]

    def rollback(self, name: str) -> RegistryEntry:
        """Restore the previous production version of ``name`` in one step.

        Archives the current production version and re-promotes the one below it
        on the promotion stack.

        Raises:
            DataError: if ``name`` has no earlier production version to restore.
        """
        index = self._load_index()
        stack = index.history.get(name, [])
        if len(stack) < 2:
            raise DataError(f"{name} has no previous production version to roll back to")
        current = stack.pop()  # archive the version currently in production
        previous = stack[-1]
        index.entries = [
            _restage(entry, name, current=current, previous=previous) for entry in index.entries
        ]
        self._save_index(index)
        logger.info("rolled %s back to %s", name, previous)
        return _find(index.entries, name, previous)  # type: ignore[return-value]

    def resolve(self, reference: str) -> RegistryEntry:
        """Resolve a reference to a registered entry.

        Accepts ``name`` (the production version), ``name@production`` /
        ``name@staging`` / ``name@archived`` (a stage; staging/archived pick the
        most recently added), or ``name@<version>`` (an exact version).

        Raises:
            DataError: if nothing matches the reference.
        """
        name, _, qualifier = reference.partition("@")
        entries = [entry for entry in self._load_index().entries if entry.name == name]
        if not entries:
            raise DataError(f"no registered model named {name!r}")
        if not qualifier or qualifier == Stage.PRODUCTION.value:
            return _by_stage(entries, Stage.PRODUCTION, reference)
        if qualifier in (Stage.STAGING.value, Stage.ARCHIVED.value):
            return _by_stage(entries, Stage(qualifier), reference)
        exact = _find(entries, name, qualifier)
        if exact is None:
            raise DataError(f"{reference} does not match any registered version")
        return exact

    def path_for(self, entry: RegistryEntry) -> Path:
        """Return the content-addressed artifact directory for ``entry``."""
        return self.root / _ARTIFACTS_DIR / entry.digest

    def entries(self) -> list[RegistryEntry]:
        """Every registered entry, in registration order."""
        return list(self._load_index().entries)

    def versions_of(self, name: str) -> list[RegistryEntry]:
        """Every registered version of ``name``, in registration order."""
        return [entry for entry in self._load_index().entries if entry.name == name]

    # --------------------------------------------------------------- internal

    def _apply_promotion(self, index: _Index, name: str, version: str) -> _Index:
        """Archive the current production of ``name`` and promote ``version``."""
        stack = index.history.setdefault(name, [])
        current = stack[-1] if stack else None
        if current == version:
            return index  # already in production; nothing to do
        index.entries = [
            _restage(entry, name, current=current, previous=version) for entry in index.entries
        ]
        stack.append(version)
        return index

    def _store_artifact(self, model_dir: Path | str, digest: str) -> None:
        """Copy the artifact into ``<root>/artifacts/<digest>/`` if not present."""
        destination = self.root / _ARTIFACTS_DIR / digest
        if destination.is_dir():
            return  # content-addressed: same digest means same bytes already stored
        destination.mkdir(parents=True, exist_ok=True)
        source = Path(model_dir)
        for name in _ARTIFACT_FILES:
            shutil.copy2(source / name, destination / name)

    def _read_sidecar(self, model_dir: Path | str) -> dict[str, Any]:
        """Read a persisted artifact's ``metadata.json`` as a dict."""
        path = Path(model_dir) / METADATA_FILENAME
        if not path.is_file():
            raise DataError(f"{Path(model_dir)} is not a model artifact (no {METADATA_FILENAME})")
        return read_json_object(path, what="model sidecar")

    def _index_path(self) -> Path:
        return self.root / REGISTRY_INDEX_NAME

    def _load_index(self) -> _Index:
        path = self._index_path()
        if not path.is_file():
            return _Index()
        data = read_json_object(path, what="registry index")
        if "entries" not in data:
            raise DataError(f"{path} is not a tulip registry index")
        return _Index.model_validate(data)

    def _save_index(self, index: _Index) -> None:
        write_sorted_json(self._index_path(), index.model_dump(mode="json"))
