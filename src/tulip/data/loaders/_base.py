"""Shared machinery for manifest-backed corpus loaders.

Most catalogued corpora have no bulk download; the researcher assembles a
local manifest (see ``docs/datasets.md`` and :mod:`tulip.data.manifest`).
:class:`ManifestBackedLoader` gives each such corpus a one-screen loader:
subclasses declare their canonical name, column mapping, and label defaults,
and inherit manifest discovery, parsing, and validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from tulip.core.exceptions import DataError
from tulip.core.interfaces import DatasetLoader
from tulip.core.types import DatasetInfo, Sample
from tulip.data.catalog import get_dataset_info
from tulip.data.manifest import ManifestColumns, read_manifest

#: Manifest file names probed (in order) when none is configured explicitly.
DEFAULT_MANIFEST_NAMES = ("manifest.csv", "manifest.tsv", "manifest.jsonl")


class ManifestBackedLoader(DatasetLoader):
    """Base class for loaders that read a locally assembled manifest.

    Subclasses set :attr:`dataset_name` (must exist in the catalog) and may
    override :attr:`columns` and :attr:`label_defaults`. The expected local
    layout is ``data/raw/<dataset_name>/manifest.{csv|tsv|jsonl}`` with audio
    paths (if any) relative to that directory.

    Args:
        manifest: Manifest path relative to ``root``, overriding the default
            ``manifest.{csv|tsv|jsonl}`` probe order.
    """

    #: Canonical catalog/registry name; subclasses must override.
    dataset_name: ClassVar[str]
    #: Column mapping passed to :func:`tulip.data.manifest.read_manifest`.
    columns: ClassVar[ManifestColumns] = ManifestColumns()
    #: Label values applied when the manifest does not provide the field.
    label_defaults: ClassVar[dict[str, str]] = {}

    def __init__(self, manifest: str | None = None) -> None:
        self._manifest = manifest

    @property
    def info(self) -> DatasetInfo:
        """Catalog metadata for this corpus."""
        return get_dataset_info(self.dataset_name)

    def load(self, root: Path) -> Iterator[Sample]:
        """Yield samples from the manifest under ``root``.

        Raises:
            DataError: if no manifest is found or it is malformed.
        """
        yield from read_manifest(
            self._resolve_manifest(root),
            columns=self.columns,
            source=self.dataset_name,
            label_defaults=self.label_defaults,
            audio_root=root,
        )

    def is_available(self, root: Path) -> bool:
        """Whether a manifest for this corpus exists under ``root``."""
        try:
            self._resolve_manifest(root)
        except DataError:
            return False
        return True

    def _resolve_manifest(self, root: Path) -> Path:
        """Locate the manifest file under ``root``."""
        if self._manifest is not None:
            path = root / self._manifest
            if not path.is_file():
                raise DataError(f"{self.dataset_name}: configured manifest not found: {path}")
            return path
        for name in DEFAULT_MANIFEST_NAMES:
            path = root / name
            if path.is_file():
                return path
        raise DataError(
            f"{self.dataset_name}: no manifest found under {root}; expected one of "
            f"{', '.join(DEFAULT_MANIFEST_NAMES)} (see docs/datasets.md for how to "
            "assemble this corpus locally)"
        )


__all__ = ["DEFAULT_MANIFEST_NAMES", "ManifestBackedLoader"]
