"""End-to-end dataset construction: load -> clean -> dedup -> split -> persist.

:class:`DatasetBuilder` turns a declarative :class:`~tulip.config.schemas.DataConfig`
into leakage-free, reproducible train/validation/test splits, and records a
build manifest (counts, class distribution, and the exact configuration) so a
published benchmark split can be audited and regenerated.

Order matters and is deliberate: deduplication runs on the *whole* corpus
before splitting, so near-duplicate texts can never straddle a split
boundary, and speaker-disjoint splitting runs last on the surviving samples.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tulip.core.exceptions import DataError
from tulip.data.cleaning import TextCleaner
from tulip.data.dedup import deduplicate_samples
from tulip.data.registry import DATASETS
from tulip.data.splitting import DatasetSplits, save_splits, speaker_disjoint_split
from tulip.labels.taxonomy import LabelLevel
from tulip.utils.io import write_json
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from tulip.config.schemas import DataConfig, SplitConfig
    from tulip.core.types import Sample

_logger = get_logger(__name__)

#: File name of the audit manifest written next to the split files.
BUILD_MANIFEST_NAME = "build_manifest.json"


class DatasetBuilder:
    """Build reproducible dataset splits from a declarative data config.

    Args:
        config: Which corpora to load and how to prepare them.
        cleaner: Text normaliser applied when ``config.clean`` is true
            (default: a :class:`TextCleaner` with standard settings --
            dialectal orthography is always preserved).
        dedup_params: Overrides forwarded to
            :func:`~tulip.data.dedup.deduplicate_samples` (e.g. ``threshold``).
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        cleaner: TextCleaner | None = None,
        dedup_params: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.cleaner = cleaner if cleaner is not None else TextCleaner()
        self.dedup_params = dict(dedup_params or {})

    def load_samples(self) -> list[Sample]:
        """Load and prepare (clean, filter, dedup) samples from every corpus.

        Each dataset entry's local root defaults to ``config.root/<name>``;
        a ``root`` key in the entry's params overrides it (absolute, or
        relative to ``config.root``). All other params go to the loader's
        constructor.

        Returns:
            Prepared samples in deterministic (config, then file) order.

        Raises:
            DataError: if a corpus is missing/malformed or nothing survives
                preparation.
        """
        samples: list[Sample] = []
        for entry in self.config.datasets:
            params = dict(entry.params)
            root = self._resolve_root(entry.name, params.pop("root", None))
            loader = DATASETS.create(entry.name, **params)
            loaded = list(loader.load(root))
            _logger.info("loaded %d samples from %s (%s)", len(loaded), entry.name, root)
            samples.extend(loaded)
        if not samples:
            raise DataError(
                "no samples were loaded from any configured dataset; check data.root "
                f"({self.config.root}) and the per-corpus layouts in docs/datasets.md"
            )
        return self._prepare(samples)

    def build(
        self,
        split: SplitConfig,
        *,
        target: LabelLevel | None = None,
        output_dir: Path | None = None,
    ) -> DatasetSplits:
        """Produce speaker-disjoint splits, optionally persisted with a manifest.

        Args:
            split: Split fractions, grouping key, stratification, seed.
            target: When given, samples lacking a label at this level are
                dropped (with a logged count) before splitting -- an
                experiment can only train on labelled data.
            output_dir: When given, writes ``train/validation/test.jsonl``
                plus ``build_manifest.json`` (counts, class distribution,
                configuration echo) into this directory.

        Returns:
            The in-memory :class:`DatasetSplits`.

        Raises:
            DataError: if no labelled samples remain or a split would be empty.
        """
        samples = self.load_samples()
        if target is not None:
            labelled = [s for s in samples if s.labels.at_level(target) is not None]
            dropped = len(samples) - len(labelled)
            if dropped:
                _logger.info(
                    "dropped %d/%d samples without a %r label", dropped, len(samples), target.value
                )
            if not labelled:
                raise DataError(
                    f"no samples carry a label at level {target.value!r}; nothing to build"
                )
            samples = labelled

        splits = speaker_disjoint_split(samples, split)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_splits(splits, output_dir)
            write_json(
                output_dir / BUILD_MANIFEST_NAME,
                self._build_manifest(splits, split, target),
            )
            _logger.info("wrote splits and %s to %s", BUILD_MANIFEST_NAME, output_dir)
        return splits

    def _resolve_root(self, name: str, override: Any) -> Path:
        """Resolve one dataset's local root directory."""
        if override is None:
            return self.config.root / name
        candidate = Path(str(override))
        return candidate if candidate.is_absolute() else self.config.root / candidate

    def _prepare(self, samples: list[Sample]) -> list[Sample]:
        """Apply cleaning, the minimum-length filter, and deduplication."""
        if self.config.clean:
            samples = list(self.cleaner.clean_samples(samples))

        min_chars = self.config.min_text_chars
        if min_chars > 0:
            before = len(samples)
            # Audio-only samples pass: the length filter is about degenerate text.
            samples = [s for s in samples if s.text is None or len(s.text) >= min_chars]
            if len(samples) != before:
                _logger.info(
                    "dropped %d samples shorter than %d characters",
                    before - len(samples),
                    min_chars,
                )

        if self.config.deduplicate:
            samples = deduplicate_samples(samples, **self.dedup_params).samples

        if not samples:
            raise DataError("no samples survived cleaning/filtering/deduplication")
        return samples

    def _build_manifest(
        self,
        splits: DatasetSplits,
        split: SplitConfig,
        target: LabelLevel | None,
    ) -> dict[str, Any]:
        """Assemble the audit manifest for a persisted build."""
        level = target if target is not None else LabelLevel.DIALECT
        distribution = {
            name: dict(
                Counter(sample.labels.at_level(level) or "__unlabelled__" for sample in samples)
            )
            for name, samples in splits.as_dict().items()
        }
        sources = dict(
            Counter(sample.source for samples in splits.as_dict().values() for sample in samples)
        )
        return {
            "sizes": splits.sizes(),
            "total": splits.total,
            "label_level": level.value,
            "class_distribution": distribution,
            "sources": sources,
            "data_config": self.config.model_dump(mode="json"),
            "split_config": split.model_dump(mode="json"),
            "cleaner": self.cleaner.config() if self.config.clean else None,
            "dedup_params": self.dedup_params if self.config.deduplicate else None,
        }


__all__ = ["BUILD_MANIFEST_NAME", "DatasetBuilder"]
