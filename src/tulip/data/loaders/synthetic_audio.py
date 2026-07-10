"""Loader for the in-process synthetic *audio* corpus (no data acquisition).

Like the text :class:`~tulip.data.loaders.synthetic.SyntheticLoader`, this loader
needs nothing acquired: it *synthesises* dialect-correlated WAV clips on demand
(see :mod:`tulip.data.synthetic_audio`) so the whole audio path is runnable end
to end out of the box. Unlike the text corpus, audio must exist on disk for the
feature extractors to decode, so ``load`` writes the clips under ``root`` (in an
``audio/`` sub-directory) as it generates them. If a manifest has been
materialised under ``root`` (e.g. via
:func:`tulip.data.synthetic_audio.write_synthetic_audio_manifest`), that
auditable copy is read instead of regenerating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tulip.core.exceptions import DataError
from tulip.core.interfaces import DatasetLoader
from tulip.data.catalog import get_dataset_info
from tulip.data.loaders._base import DEFAULT_MANIFEST_NAMES
from tulip.data.manifest import read_manifest
from tulip.data.registry import DATASETS
from tulip.data.synthetic_audio import SOURCE, AudioSyntheticSpec, generate_audio_corpus
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from tulip.core.types import DatasetInfo, Sample

_logger = get_logger(__name__)


@DATASETS.register("synthetic_audio")
class SyntheticAudioLoader(DatasetLoader):
    """Generate (or read back) the synthetic audio reference corpus.

    The generation knobs mirror
    :class:`~tulip.data.synthetic_audio.AudioSyntheticSpec` and arrive through
    :class:`~tulip.config.schemas.ComponentConfig` ``params`` (which
    :meth:`~tulip.data.builder.DatasetBuilder.load_samples` forwards to
    ``DATASETS.create`` after popping ``root``), so no config-schema change is
    needed to configure the corpus from a YAML experiment.

    Args:
        n_speakers_per_dialect: Speakers per class (>= 2).
        samples_per_speaker: Clips per speaker.
        dialects: Class keys to include (``None`` = all).
        duration_s: Clip duration in seconds.
        sample_rate: Output sample rate in Hz.
        jitter: Per-speaker relative jitter on F0 and formants.
        seed: Generator seed (fixes the whole corpus).
        manifest: Optional manifest file name relative to ``root``; when set,
            it must exist and is read instead of generating.
    """

    auto_downloadable: ClassVar[bool] = False

    acquisition: ClassVar[str] = (
        "none required: this corpus is synthesised in-process by "
        "tulip.data.synthetic_audio; point any experiment at the "
        "'synthetic_audio' dataset and it materialises WAV clips with zero data "
        "acquisition. See docs/datasets.md."
    )

    def __init__(
        self,
        *,
        n_speakers_per_dialect: int = 6,
        samples_per_speaker: int = 6,
        dialects: Sequence[str] | None = None,
        duration_s: float = 0.8,
        sample_rate: int = 16_000,
        jitter: float = 0.06,
        seed: int = 7,
        manifest: str | None = None,
    ) -> None:
        self._spec = AudioSyntheticSpec(
            n_speakers_per_dialect=n_speakers_per_dialect,
            samples_per_speaker=samples_per_speaker,
            dialects=tuple(dialects) if dialects is not None else None,
            duration_s=duration_s,
            sample_rate=sample_rate,
            jitter=jitter,
            seed=seed,
        )
        self._manifest = manifest

    @property
    def info(self) -> DatasetInfo:
        """Catalog metadata for the synthetic audio corpus."""
        return get_dataset_info(SOURCE)

    def is_available(self, root: Path) -> bool:
        """Always available: the corpus is synthesised on demand (bigos precedent)."""
        del root
        return True

    def load(self, root: Path) -> Iterator[Sample]:
        """Read a materialised manifest under ``root`` if present, else generate.

        Generation writes the WAV clips under ``root/audio`` and yields the
        corresponding samples.

        Raises:
            DataError: if a manifest name was configured but is not present.
        """
        manifest_path = self._resolve_manifest(root)
        if manifest_path is not None:
            _logger.info("loading synthetic audio corpus from manifest %s", manifest_path)
            yield from read_manifest(manifest_path, source=SOURCE)
            return
        _logger.info("generating synthetic audio corpus in-process (seed=%d)", self._spec.seed)
        yield from generate_audio_corpus(self._spec, root)

    def _resolve_manifest(self, root: Path) -> Path | None:
        """Return the manifest to read, or ``None`` to generate on demand."""
        if self._manifest is not None:
            path = root / self._manifest
            if not path.is_file():
                raise DataError(f"synthetic_audio: configured manifest not found: {path}")
            return path
        for name in DEFAULT_MANIFEST_NAMES:
            candidate = root / name
            if candidate.is_file():
                return candidate
        return None


__all__ = ["SyntheticAudioLoader"]
