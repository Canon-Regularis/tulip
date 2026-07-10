"""Loader for the in-process synthetic reference corpus (no data acquisition).

Unlike the other loaders, this one needs nothing on disk: it *generates* a
linguistically-grounded corpus on demand (see :mod:`tulip.data.synthetic`), so
the whole toolkit is runnable end-to-end out of the box. If a manifest has been
materialised under ``root`` (e.g. via
:func:`tulip.data.synthetic.write_synthetic_manifest`), that auditable copy is
read instead of regenerating -- mirroring the on-demand precedent set by the
BIGOS/Common Voice loaders that also override :meth:`is_available`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tulip.core.exceptions import DataError
from tulip.core.interfaces import DatasetLoader
from tulip.data.catalog import get_dataset_info
from tulip.data.loaders._base import DEFAULT_MANIFEST_NAMES
from tulip.data.manifest import read_manifest
from tulip.data.registry import DATASETS
from tulip.data.synthetic import SOURCE, SyntheticSpec, generate_corpus
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from tulip.core.types import DatasetInfo, Sample

_logger = get_logger(__name__)


@DATASETS.register("synthetic")
class SyntheticLoader(DatasetLoader):
    """Generate (or read back) the synthetic reference corpus.

    The generation knobs mirror :class:`~tulip.data.synthetic.SyntheticSpec`
    and arrive through :class:`~tulip.config.schemas.ComponentConfig` ``params``
    (which :meth:`~tulip.data.builder.DatasetBuilder.load_samples` forwards to
    ``DATASETS.create`` after popping ``root``), so no config-schema change is
    needed to configure the corpus from a YAML experiment.

    Args:
        n_speakers_per_dialect: Speakers per class (>= 2).
        samples_per_speaker: Texts per speaker.
        dialects: Lexicon keys to include (``None`` = all).
        include_standard: Add the ``standard`` negative class.
        noise_level: Cross-class marker leakage probability.
        marker_dropout: Probability a sample carries no lexical marker, which
            gives the task an irreducible error floor (see the module docs).
        seed: Generator seed (fixes the whole corpus).
        manifest: Optional manifest file name relative to ``root``; when set,
            it must exist and is read instead of generating.
    """

    auto_downloadable: ClassVar[bool] = False

    acquisition: ClassVar[str] = (
        "none required: this corpus is generated in-process by "
        "tulip.data.synthetic; point any experiment at the 'synthetic' dataset "
        "and it materialises with zero data acquisition. Run "
        "`tulip data synthesize` to write an auditable copy. See docs/datasets.md."
    )

    def __init__(
        self,
        *,
        n_speakers_per_dialect: int = 8,
        samples_per_speaker: int = 12,
        dialects: Sequence[str] | None = None,
        include_standard: bool = True,
        noise_level: float = 0.10,
        marker_dropout: float = 0.20,
        seed: int = 7,
        manifest: str | None = None,
    ) -> None:
        self._spec = SyntheticSpec(
            n_speakers_per_dialect=n_speakers_per_dialect,
            samples_per_speaker=samples_per_speaker,
            dialects=tuple(dialects) if dialects is not None else None,
            include_standard=include_standard,
            noise_level=noise_level,
            marker_dropout=marker_dropout,
            seed=seed,
        )
        self._manifest = manifest

    @property
    def info(self) -> DatasetInfo:
        """Catalog metadata for the synthetic corpus."""
        return get_dataset_info(SOURCE)

    def is_available(self, root: Path) -> bool:
        """Always available: the corpus is generated on demand (bigos precedent)."""
        del root
        return True

    def load(self, root: Path) -> Iterator[Sample]:
        """Read a materialised manifest under ``root`` if present, else generate.

        Raises:
            DataError: if a manifest name was configured but is not present.
        """
        manifest_path = self._resolve_manifest(root)
        if manifest_path is not None:
            _logger.info("loading synthetic corpus from manifest %s", manifest_path)
            yield from read_manifest(manifest_path, source=SOURCE)
            return
        _logger.info("generating synthetic corpus in-process (seed=%d)", self._spec.seed)
        yield from generate_corpus(self._spec)

    def _resolve_manifest(self, root: Path) -> Path | None:
        """Return the manifest to read, or ``None`` to generate on demand."""
        if self._manifest is not None:
            path = root / self._manifest
            if not path.is_file():
                raise DataError(f"synthetic: configured manifest not found: {path}")
            return path
        for name in DEFAULT_MANIFEST_NAMES:
            candidate = root / name
            if candidate.is_file():
                return candidate
        return None


__all__ = ["SyntheticLoader"]
