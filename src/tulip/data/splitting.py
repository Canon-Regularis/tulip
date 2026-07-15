"""Speaker-disjoint, stratified train/validation/test splitting.

The cardinal rule for dialect benchmarks: **no speaker appears in more than
one split**. Dialect classifiers trivially learn speaker identity (voice,
idiolect, favourite words), so speaker overlap turns dialect evaluation into
speaker re-identification. Splitting therefore assigns whole *groups*
(default: ``speaker_id``) to splits, never individual samples.

Assignment is greedy and label-aware: groups are processed largest-first (in
a seed-shuffled deterministic order) and each goes to the split with the
largest remaining deficit for the group's majority label at the configured
stratification level. Exact stratification is impossible under group
constraints; the greedy pass gets close in practice and is fully
deterministic for a given seed. A rescue pass guarantees that no
positive-fraction split is left empty while unassigned groups remain.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tulip.core.exceptions import DataError
from tulip.core.types import Sample
from tulip.utils.io import read_jsonl, write_jsonl
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tulip.config.schemas import SplitConfig

_logger = get_logger(__name__)

_SPLIT_NAMES = ("train", "validation", "test")
_UNLABELLED = "__unlabelled__"


@dataclass(frozen=True)
class DatasetSplits:
    """Named train/validation/test partitions of a sample collection."""

    train: list[Sample]
    validation: list[Sample]
    test: list[Sample]

    def as_dict(self) -> dict[str, list[Sample]]:
        """Return ``{split_name: samples}`` in canonical order."""
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def sizes(self) -> dict[str, int]:
        """Return the number of samples per split."""
        return {name: len(samples) for name, samples in self.as_dict().items()}

    @property
    def total(self) -> int:
        """Total number of samples across all splits."""
        return len(self.train) + len(self.validation) + len(self.test)


def speaker_disjoint_split(samples: Iterable[Sample], config: SplitConfig) -> DatasetSplits:
    """Partition samples into group-disjoint train/validation/test splits.

    Args:
        samples: The samples to split. Order does not affect which split a
            group lands in (grouping is by key), but within-split sample
            order follows input order.
        config: Fractions, grouping attribute, stratification level, seed.

    Returns:
        :class:`DatasetSplits` with zero group overlap between splits.

    Raises:
        DataError: if there are no samples, a sample lacks the grouping key,
            or any split with a positive configured fraction ends up empty
            (too few distinct groups).
    """
    sample_list = list(samples)
    if not sample_list:
        raise DataError("cannot split an empty sample collection")

    groups = _group_indices(sample_list, config.group_by)
    group_labels = _majority_labels(sample_list, groups, config)
    fractions = {"train": config.train, "validation": config.validation, "test": config.test}

    order = _deterministic_group_order(groups, config.seed)
    assignment = _greedy_assign(order, groups, group_labels, fractions, len(sample_list))

    split_samples: dict[str, list[Sample]] = {name: [] for name in _SPLIT_NAMES}
    for sample in sample_list:
        group = _group_key(sample, config.group_by)
        split_samples[assignment[group]].append(sample)

    for name in _SPLIT_NAMES:
        if fractions[name] > 0.0 and not split_samples[name]:
            raise DataError(
                f"split {name!r} (fraction {fractions[name]:.2f}) received no samples; "
                f"only {len(groups)} distinct {config.group_by!r} group(s) are available "
                "-- more groups (speakers) are needed for a disjoint split"
            )

    splits = DatasetSplits(**split_samples)
    _logger.info(
        "speaker-disjoint split (%d groups, seed=%d): %s",
        len(groups),
        config.seed,
        splits.sizes(),
    )
    return splits


def _group_key(sample: Sample, group_by: str) -> str:
    """Extract and validate the grouping key for one sample."""
    if not hasattr(sample, group_by):
        raise DataError(f"samples have no attribute {group_by!r} to group by")
    value = getattr(sample, group_by)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise DataError(
            f"sample {sample.id!r} has no {group_by!r}; loaders must synthesise a "
            "surrogate speaker ID so splits stay leakage-free"
        )
    return str(value)


def _group_indices(samples: Sequence[Sample], group_by: str) -> dict[str, list[int]]:
    """Map each group key to the indices of its samples."""
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(_group_key(sample, group_by), []).append(index)
    return groups


def _majority_labels(
    samples: Sequence[Sample],
    groups: dict[str, list[int]],
    config: SplitConfig,
) -> dict[str, str]:
    """Determine each group's stratification label (majority vote).

    Groups are almost always label-pure (a speaker speaks one dialect); the
    majority vote handles noisy corpora deterministically (count, then
    lexicographic tie-break).
    """
    level = config.stratify_by
    labels: dict[str, str] = {}
    for group, indices in groups.items():
        if level is None:
            labels[group] = _UNLABELLED
            continue
        counts = Counter(samples[i].labels.at_level(level) or _UNLABELLED for i in indices)
        labels[group] = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return labels


def _deterministic_group_order(groups: dict[str, list[int]], seed: int) -> list[str]:
    """Order groups largest-first with a seeded shuffle breaking size ties."""
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    shuffled = [names[i] for i in rng.permutation(len(names))]
    return sorted(shuffled, key=lambda g: -len(groups[g]))  # stable: keeps shuffle order


def _greedy_assign(
    order: Sequence[str],
    groups: dict[str, list[int]],
    group_labels: dict[str, str],
    fractions: dict[str, float],
    total: int,
) -> dict[str, str]:
    """Assign each group to a split, chasing per-label sample deficits."""
    label_totals: Counter[str] = Counter()
    for group, indices in groups.items():
        label_totals[group_labels[group]] += len(indices)

    per_label: dict[str, Counter[str]] = {name: Counter() for name in _SPLIT_NAMES}
    per_split: dict[str, int] = dict.fromkeys(_SPLIT_NAMES, 0)
    assigned_groups: dict[str, int] = dict.fromkeys(_SPLIT_NAMES, 0)
    assignment: dict[str, str] = {}

    for position, group in enumerate(order):
        label = group_labels[group]
        weight = len(groups[group])

        remaining = len(order) - position
        empty_positive = [
            name for name in _SPLIT_NAMES if fractions[name] > 0.0 and assigned_groups[name] == 0
        ]
        if empty_positive and remaining <= len(empty_positive):
            # Rescue pass: exactly enough groups remain to cover the still-empty
            # splits, so fill them (largest fraction first) instead of chasing
            # deficits: an empty split is worse than an off-target one.
            target = max(empty_positive, key=lambda name: fractions[name])
        else:
            target = max(
                _SPLIT_NAMES,
                key=lambda name: (
                    fractions[name] * label_totals[label] - per_label[name][label],
                    fractions[name] * total - per_split[name],
                    -_SPLIT_NAMES.index(name),
                ),
            )

        assignment[group] = target
        per_label[target][label] += weight
        per_split[target] += weight
        assigned_groups[target] += 1

    return assignment


def save_splits(splits: DatasetSplits, directory: Path | str) -> dict[str, Path]:
    """Persist splits as one JSONL file per split under ``directory``.

    Returns:
        ``{split_name: written_path}`` (all three files are always written,
        even when empty, so a directory is unambiguously a complete split).
    """
    directory = Path(directory)
    paths: dict[str, Path] = {}
    for name, samples in splits.as_dict().items():
        path = directory / f"{name}.jsonl"
        write_jsonl(path, (sample.model_dump(mode="json") for sample in samples))
        paths[name] = path
    return paths


def load_splits(directory: Path | str) -> DatasetSplits:
    """Load splits previously written by :func:`save_splits`.

    Raises:
        DataError: if any of the three split files is missing or malformed.
    """
    directory = Path(directory)
    loaded: dict[str, list[Sample]] = {}
    for name in _SPLIT_NAMES:
        path = directory / f"{name}.jsonl"
        if not path.is_file():
            raise DataError(f"split file not found: {path}")
        try:
            loaded[name] = [Sample.model_validate(record) for record in read_jsonl(path)]
        except ValueError as exc:
            raise DataError(f"invalid split file {path}: {exc}") from exc
    return DatasetSplits(**loaded)


__all__ = ["DatasetSplits", "load_splits", "save_splits", "speaker_disjoint_split"]
