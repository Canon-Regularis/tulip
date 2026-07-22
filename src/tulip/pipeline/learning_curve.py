"""Learning curves: how metric quality scales with training-set size.

A benchmark number at one corpus size says nothing about what more annotation
would buy. This trains the experiment's model on nested, stratified fractions
of the training split and evaluates every point on the identical held-out test
split, so the curve isolates training-set size as the only moving part.

The fractions are nested: the samples in a smaller fraction are always a subset
of every larger one. A fresh per-class order is drawn once from the seed and
each fraction takes a prefix, which removes subset lottery noise from the curve
and makes the whole report deterministic and byte-stable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import save_report
from tulip.core.exceptions import ConfigurationError
from tulip.utils.logging import get_logger
from tulip.utils.seed import set_global_seed

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tulip.config.schemas import ExperimentConfig
    from tulip.core.types import Sample

__all__ = [
    "DEFAULT_FRACTIONS",
    "LearningCurvePoint",
    "LearningCurveReport",
    "learning_curve",
]

_logger = get_logger(__name__)

#: Fractions of the training split evaluated by default.
DEFAULT_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)

#: Stored floats are rounded to this many digits so a saved report is
#: byte-identical when the content is, mirroring the other rigor reports.
LEARNING_CURVE_FLOAT_DIGITS = 6


class LearningCurvePoint(BaseModel):
    """One training-set size and the test metrics it reaches."""

    model_config = ConfigDict(frozen=True)

    fraction: float = Field(gt=0.0, le=1.0)
    n_train: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)


class LearningCurveReport(BaseModel):
    """The full curve for one experiment, smallest fraction first."""

    model_config = ConfigDict(frozen=True)

    model: str
    target: str
    seed: int
    points: tuple[LearningCurvePoint, ...]

    def to_markdown(self) -> str:
        """Render the curve as a markdown table."""
        from tulip.evaluation._format import format_metric, markdown_table

        rows = [
            (
                f"{point.fraction:.2f}",
                str(point.n_train),
                format_metric(point.accuracy),
                format_metric(point.f1_macro),
            )
            for point in self.points
        ]
        title = f"# Learning curve: {self.model} ({self.target})"
        headers = ("Fraction", "Train", "Accuracy", "F1 (macro)")
        return f"{title}\n\n{markdown_table(headers, rows)}"

    def save(self, path: Path | str) -> None:
        """Write the report as deterministic JSON (sorted keys, rounded floats)."""
        save_report(self, path, digits=LEARNING_CURVE_FLOAT_DIGITS)


def learning_curve(
    config: ExperimentConfig,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    seed: int | None = None,
) -> LearningCurveReport:
    """Train on nested fractions of the training split and score each on test.

    Args:
        config: The experiment declaration (data, features, model, target).
        fractions: Training fractions to evaluate; each in ``(0, 1]``.
        seed: Subsampling seed; defaults to the config's seed.

    Returns:
        A :class:`LearningCurveReport`, smallest fraction first.

    Raises:
        ConfigurationError: if ``fractions`` is empty or holds a value outside
            ``(0, 1]``.
        DataError: if a fraction leaves fewer than two classes to fit on
            (raised by the classifier's fit validation).
    """
    ordered_fractions = _validate_fractions(fractions)
    subsample_seed = config.seed if seed is None else seed

    from tulip.data.builder import DatasetBuilder
    from tulip.pipeline.experiment import build_classifier, evaluate_samples

    splits = DatasetBuilder(config.data).build(config.split, target=config.target)
    by_class = _class_orders(splits.train, config, subsample_seed)

    points: list[LearningCurvePoint] = []
    for fraction in ordered_fractions:
        subset = _take_fraction(by_class, fraction)
        set_global_seed(config.seed)
        classifier = build_classifier(config)
        classifier.fit(subset)
        report = evaluate_samples(classifier, splits.test, name=f"fraction-{fraction}")
        points.append(
            LearningCurvePoint(
                fraction=fraction,
                n_train=len(subset),
                accuracy=report.accuracy,
                f1_macro=report.f1_macro,
            )
        )
        _logger.info(
            "learning curve: fraction %.2f -> %d train samples, f1_macro %.4f",
            fraction,
            len(subset),
            report.f1_macro,
        )

    return LearningCurveReport(
        model=config.model.name,
        target=config.target.value,
        seed=subsample_seed,
        points=tuple(points),
    )


def _validate_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    """Sorted unique fractions, each in ``(0, 1]``."""
    if not fractions:
        raise ConfigurationError("learning_curve needs at least one fraction")
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ConfigurationError(f"fractions must be in (0, 1], got {fraction}")
    return tuple(sorted(set(fractions)))


def _class_orders(
    train: Sequence[Sample], config: ExperimentConfig, seed: int
) -> dict[str, list[Sample]]:
    """One seeded shuffle per class; every fraction takes a prefix of these."""
    grouped: dict[str, list[Sample]] = {}
    for sample in train:
        label = sample.labels.at_level(config.target)
        if label is None:
            continue  # unlabelled at the target level; the classifier drops it at fit
        grouped.setdefault(str(label), []).append(sample)
    rng = np.random.default_rng(seed)
    ordered: dict[str, list[Sample]] = {}
    for label in sorted(grouped):
        members = grouped[label]
        permutation = rng.permutation(len(members))
        ordered[label] = [members[index] for index in permutation]
    return ordered


def _take_fraction(by_class: dict[str, list[Sample]], fraction: float) -> list[Sample]:
    """A stratified prefix: at least one sample of every class, nested by design."""
    subset: list[Sample] = []
    for label in sorted(by_class):
        members = by_class[label]
        count = max(1, math.ceil(fraction * len(members)))
        subset.extend(members[:count])
    return subset
