"""Closed-loop active learning: acquire, label, retrain, repeat.

:mod:`tulip.pipeline.active` ranks an unlabeled pool but stops there; turning a
ranking into labels and folding them back into training was left as a manual
step. This module closes that loop as a *simulation*: it starts from a small
seed set, and each round asks the acquisition strategy which pool samples to
label next, reveals their gold labels through an oracle, refits, and scores the
identical held-out test split. The result is a learning curve indexed by
annotation budget instead of a fixed fraction, which is what tells you whether a
strategy buys accuracy faster than labeling at random.

The oracle is the pool's own gold labels: the pool is fully labeled, the loop
merely hides those labels until a sample is acquired. That is the standard way
to measure an acquisition strategy offline, and it needs no human in the loop.
Pass ``strategy="random"`` for the baseline every real strategy must beat.

Everything reuses existing blocks: splits from
:class:`~tulip.data.builder.DatasetBuilder`, ranking from
:func:`~tulip.pipeline.active.rank_for_labeling`, the classifier from
:func:`~tulip.pipeline.experiment.build_classifier`, and scoring from
:func:`~tulip.pipeline.experiment.evaluate_samples`. The run is fully seeded:
the stratified seed set, the per-round acquisition, and every refit are
deterministic, so a saved :class:`ActiveLoopReport` is byte-stable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import round_floats, write_sorted_json
from tulip.core.exceptions import ConfigurationError
from tulip.utils.logging import get_logger
from tulip.utils.seed import set_global_seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.config.schemas import ExperimentConfig
    from tulip.core.types import Sample

__all__ = [
    "ACTIVE_LOOP_FLOAT_DIGITS",
    "ActiveLoopPoint",
    "ActiveLoopReport",
    "active_learning_loop",
]

_logger = get_logger(__name__)

#: Name of the random-acquisition baseline handled inside the loop (it needs a
#: seed, which the pure-scoring acquisition strategies do not carry).
RANDOM_STRATEGY = "random"

#: Stored floats are rounded to this many digits so a saved report is
#: byte-identical when the content is, matching the other rigor reports.
ACTIVE_LOOP_FLOAT_DIGITS = 6


class ActiveLoopPoint(BaseModel):
    """One acquisition round: how much is labeled, and the test metrics it reaches."""

    model_config = ConfigDict(frozen=True)

    round: int = Field(ge=0)
    n_labeled: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)


class ActiveLoopReport(BaseModel):
    """The learning curve of one acquisition strategy, round 0 (seed) first."""

    model_config = ConfigDict(frozen=True)

    strategy: str
    model: str
    target: str
    seed: int
    seed_size: int
    batch_size: int
    points: tuple[ActiveLoopPoint, ...]

    def to_markdown(self) -> str:
        """Render the acquisition curve as a markdown table."""
        from tulip.evaluation._format import format_metric, markdown_table

        rows = [
            (
                str(point.round),
                str(point.n_labeled),
                format_metric(point.accuracy),
                format_metric(point.f1_macro),
            )
            for point in self.points
        ]
        title = f"# Active learning: {self.model} ({self.target}), strategy={self.strategy}"
        headers = ("Round", "Labeled", "Accuracy", "F1 (macro)")
        return f"{title}\n\n{markdown_table(headers, rows)}"

    def save(self, path: Path | str) -> None:
        """Write the report as deterministic JSON (sorted keys, rounded floats)."""
        payload = round_floats(self.model_dump(mode="json"), ACTIVE_LOOP_FLOAT_DIGITS)
        write_sorted_json(Path(path), payload)


def active_learning_loop(
    config: ExperimentConfig,
    *,
    strategy: str = "entropy",
    seed_size: int = 20,
    batch_size: int = 20,
    rounds: int = 5,
    seed: int | None = None,
) -> ActiveLoopReport:
    """Simulate acquire -> label -> retrain over the config's training split.

    The training split is the pool; a stratified seed set is labeled up front,
    and each round labels ``batch_size`` more, chosen by ``strategy`` (or drawn
    at random for ``strategy="random"``). Every point is scored on the identical
    held-out test split.

    Args:
        config: The experiment declaration (data, features, model, target). Its
            training split is the pool; its test split scores every round.
        strategy: A registered acquisition strategy name, or ``"random"`` for
            the baseline. Dialect-aware strategies need a text task.
        seed_size: Samples labeled before round 1; drawn stratified by class so
            every class is present (a fit needs at least two).
        batch_size: Samples acquired and labeled each round.
        rounds: Maximum acquisition rounds; the loop stops early once the pool
            is exhausted.
        seed: Seed for the seed-set draw and random acquisition; defaults to the
            config's seed.

    Returns:
        An :class:`ActiveLoopReport`, round 0 (seed only) first.

    Raises:
        ConfigurationError: if ``seed_size``, ``batch_size``, or ``rounds`` is
            not positive, or ``seed_size`` exceeds the pool.
        DataError: if the seed set leaves fewer than two classes to fit on
            (raised by the classifier's fit validation).
    """
    _validate_params(seed_size=seed_size, batch_size=batch_size, rounds=rounds)
    _validate_strategy(strategy)
    loop_seed = config.seed if seed is None else seed

    from tulip.data.builder import DatasetBuilder
    from tulip.pipeline.experiment import build_classifier, evaluate_samples

    splits = DatasetBuilder(config.data).build(config.split, target=config.target)
    pool = list(splits.train)
    if seed_size > len(pool):
        raise ConfigurationError(
            f"seed_size ({seed_size}) exceeds the training pool ({len(pool)} samples)"
        )

    rng = np.random.default_rng(loop_seed)
    labeled, remaining = _stratified_seed(pool, config, seed_size, rng)

    points: list[ActiveLoopPoint] = []
    for round_index in range(rounds + 1):
        if round_index > 0:
            if not remaining:
                break
            acquired = _acquire(
                config, labeled, remaining, strategy=strategy, batch_size=batch_size, rng=rng
            )
            if not acquired:
                break
            acquired_ids = {sample.id for sample in acquired}
            labeled = [*labeled, *acquired]
            remaining = [sample for sample in remaining if sample.id not in acquired_ids]

        set_global_seed(config.seed)
        classifier = build_classifier(config)
        classifier.fit(labeled)
        report = evaluate_samples(classifier, splits.test, name=f"round-{round_index}")
        points.append(
            ActiveLoopPoint(
                round=round_index,
                n_labeled=len(labeled),
                accuracy=report.accuracy,
                f1_macro=report.f1_macro,
            )
        )
        _logger.info(
            "active loop [%s]: round %d, %d labeled, f1_macro %.4f",
            strategy,
            round_index,
            len(labeled),
            report.f1_macro,
        )

    return ActiveLoopReport(
        strategy=strategy,
        model=config.model.name,
        target=config.target.value,
        seed=loop_seed,
        seed_size=seed_size,
        batch_size=batch_size,
        points=tuple(points),
    )


def _validate_params(*, seed_size: int, batch_size: int, rounds: int) -> None:
    """Reject non-positive loop parameters up front."""
    if seed_size < 1:
        raise ConfigurationError(f"seed_size must be >= 1, got {seed_size}")
    if batch_size < 1:
        raise ConfigurationError(f"batch_size must be >= 1, got {batch_size}")
    if rounds < 1:
        raise ConfigurationError(f"rounds must be >= 1, got {rounds}")


def _validate_strategy(strategy: str) -> None:
    """Fail fast on an unknown strategy, before any splits are built or fit."""
    if strategy == RANDOM_STRATEGY:
        return
    from tulip.pipeline.active import STRATEGIES

    if strategy not in set(STRATEGIES.names()):
        options = ", ".join([*STRATEGIES.names(), RANDOM_STRATEGY])
        raise ConfigurationError(
            f"unknown acquisition strategy {strategy!r}; choose from: {options}"
        )


def _stratified_seed(
    pool: Sequence[Sample], config: ExperimentConfig, seed_size: int, rng: np.random.Generator
) -> tuple[list[Sample], list[Sample]]:
    """Split the pool into a stratified labeled seed set and the remainder.

    One seeded shuffle per class, then a round-robin over classes takes one at a
    time until ``seed_size`` is reached, so every class is represented before any
    class gets a second sample. That guarantees at least two classes in the seed
    (a fit needs it) whenever the pool has them.
    """
    by_class: dict[str, list[Sample]] = {}
    for sample in pool:
        label = str(sample.labels.at_level(config.target))
        by_class.setdefault(label, []).append(sample)
    order: dict[str, list[Sample]] = {}
    for label in sorted(by_class):
        members = by_class[label]
        permutation = rng.permutation(len(members))
        order[label] = [members[index] for index in permutation]

    seed_ids: set[str] = set()
    seed_samples: list[Sample] = []
    cursors = dict.fromkeys(order, 0)
    # Round-robin over classes in sorted order, deterministic given the shuffle.
    while len(seed_samples) < seed_size:
        progressed = False
        for label in sorted(order):
            if len(seed_samples) >= seed_size:
                break
            cursor = cursors[label]
            if cursor < len(order[label]):
                sample = order[label][cursor]
                cursors[label] = cursor + 1
                seed_samples.append(sample)
                seed_ids.add(sample.id)
                progressed = True
        if not progressed:  # pragma: no cover - guarded by seed_size <= len(pool)
            break
    remaining = [sample for sample in pool if sample.id not in seed_ids]
    return seed_samples, remaining


def _acquire(
    config: ExperimentConfig,
    labeled: Sequence[Sample],
    remaining: Sequence[Sample],
    *,
    strategy: str,
    batch_size: int,
    rng: np.random.Generator,
) -> list[Sample]:
    """Pick the next batch to label: a random draw, or an acquisition ranking.

    The random baseline is drawn here because it needs a seed the pure-scoring
    strategies do not carry. Every other strategy fits a fresh classifier on the
    current labeled set and ranks the remaining pool by
    :func:`~tulip.pipeline.active.rank_for_labeling`.
    """
    if strategy == RANDOM_STRATEGY:
        take = min(batch_size, len(remaining))
        chosen = rng.permutation(len(remaining))[:take]
        return [remaining[int(index)] for index in sorted(chosen)]

    from tulip.pipeline.active import rank_for_labeling
    from tulip.pipeline.experiment import build_classifier

    set_global_seed(config.seed)
    ranker = build_classifier(config)
    ranker.fit(labeled)
    candidates = rank_for_labeling(ranker, remaining, strategy=strategy, budget=batch_size)
    by_id = {sample.id: sample for sample in remaining}
    return [by_id[candidate.sample_id] for candidate in candidates]
