"""The generic ``manifest`` loader for ad-hoc, locally assembled corpora."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tulip.core.types import DatasetInfo, Sample
from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.manifest import ManifestColumns, read_manifest
from tulip.data.registry import DATASETS

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path


@DATASETS.register("manifest")
class GenericManifestLoader(ManifestBackedLoader):
    """Load any corpus described by a local CSV/TSV/JSONL manifest.

    This is the escape hatch for corpora without a dedicated loader: point
    it at a directory containing ``manifest.{csv|tsv|jsonl}`` (or pass an
    explicit ``manifest`` name) and map your column names onto Sample
    fields. Expected layout::

        data/raw/<anything>/
            manifest.csv        # one row per sample; see tulip.data.manifest
            clips/...           # optional audio, referenced relatively

    Args:
        manifest: Manifest file name relative to ``root`` (default: probe
            ``manifest.csv``/``.tsv``/``.jsonl``).
        columns: Sample-field -> column-name mapping; explicitly mapped
            columns must exist in the file.
        label_defaults: Label values applied to rows that lack them (e.g.
            ``{"dialect": "kurpie"}`` for a single-dialect collection).
        source: Value recorded in ``Sample.source`` (default ``"manifest"``).
    """

    dataset_name = "manifest"
    acquisition: ClassVar[str] = (
        "manual by definition: this generic loader reads whatever corpus you "
        "assemble yourself into a manifest directory (see docs/datasets.md)"
    )

    def __init__(
        self,
        manifest: str | None = None,
        *,
        columns: Mapping[str, str | None] | None = None,
        label_defaults: Mapping[str, str] | None = None,
        source: str = "manifest",
    ) -> None:
        super().__init__(manifest)
        self._columns = ManifestColumns(**dict(columns)) if columns else ManifestColumns()
        self._label_defaults = dict(label_defaults or {})
        self._source = source

    @property
    def info(self) -> DatasetInfo:
        """Generic metadata (this loader is not tied to one catalogued corpus)."""
        return DatasetInfo(
            name="manifest",
            description=(
                "Generic loader for locally assembled manifest corpora "
                "(CSV/TSV/JSONL; see tulip.data.manifest)."
            ),
            tier=4,
        )

    def load(self, root: Path) -> Iterator[Sample]:
        """Yield samples from the configured manifest under ``root``."""
        yield from read_manifest(
            self._resolve_manifest(root),
            columns=self._columns,
            source=self._source,
            label_defaults=self._label_defaults,
            audio_root=root,
        )


__all__ = ["GenericManifestLoader"]
