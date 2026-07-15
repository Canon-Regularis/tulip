"""Run a robustness sweep: score one model as its inputs are perturbed.

The sweep trains a classifier once on clean training data, scores it on the
clean test split for the baseline, then re-scores it on the test split perturbed
at each level of each perturbation. Reusing the training and evaluation
machinery keeps the numbers comparable with the rest of the benchmark; the
pipeline imports are lazy, matching :mod:`tulip.evaluation.cross_corpus`, so
there is no import cycle.

Everything is seeded. Each perturbation draws from a stream derived from
``(perturbation seed, level index)``, and samples are iterated in fixed order,
so the report regenerates byte for byte.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tulip.robustness.registry import PERTURBATIONS
from tulip.robustness.report import RobustnessCell, RobustnessCurve, RobustnessReport
from tulip.utils.logging import get_logger
from tulip.utils.seed import set_global_seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.config.schemas import ExperimentConfig
    from tulip.core.types import Sample
    from tulip.robustness.perturbations import Perturbation
    from tulip.robustness.report import PerturbationConfig

__all__ = ["perturb_samples", "run_robustness"]

_logger = get_logger(__name__)


def perturb_samples(
    samples: Sequence[Sample],
    perturbation: Perturbation,
    level: float,
    *,
    seed: int | Sequence[int],
) -> list[Sample]:
    """Return copies of ``samples`` with their text perturbed at ``level``.

    One rng drives the whole batch, consumed in sample order, so the result is
    deterministic given ``seed``. Samples without text (audio only) pass through
    unchanged.
    """
    rng = np.random.default_rng(seed)
    perturbed: list[Sample] = []
    for sample in samples:
        if sample.text is None:
            perturbed.append(sample)
            continue
        new_text = perturbation.perturb(sample.text, level=level, rng=rng)
        perturbed.append(sample.model_copy(update={"text": new_text}))
    return perturbed


def run_robustness(
    config: ExperimentConfig,
    *,
    perturbations: Sequence[PerturbationConfig],
    samples: Sequence[Sample] | None = None,
) -> RobustnessReport:
    """Train once, then score the model across perturbations and levels.

    Args:
        config: The experiment declaration (data, features, model, target, split).
        perturbations: The perturbations to sweep; each carries its own levels
            and seed.
        samples: Pre-loaded samples to split and use; loaded and prepared from
            ``config.data`` when omitted.

    Returns:
        A :class:`RobustnessReport` with the clean baseline and one curve per
        perturbation.

    Raises:
        DataError: if the data yields no usable train or test split.
    """
    from tulip.pipeline.experiment import build_classifier, evaluate_samples

    set_global_seed(config.seed)
    train, test = _train_test(config, samples)

    classifier = build_classifier(config)
    classifier.fit(train)
    clean = evaluate_samples(classifier, test, name="clean")
    baseline = RobustnessCell(
        perturbation="clean",
        level=0.0,
        n=clean.n_samples,
        accuracy=clean.accuracy,
        f1_macro=clean.f1_macro,
    )

    curves: list[RobustnessCurve] = []
    for spec in perturbations:
        perturbation = PERTURBATIONS.create(spec.name, **spec.params)
        cells: list[RobustnessCell] = []
        for level_index, level in enumerate(spec.levels):
            perturbed = perturb_samples(test, perturbation, level, seed=(spec.seed, level_index))
            report = evaluate_samples(classifier, perturbed, name=f"{spec.name}@{level}")
            cells.append(
                RobustnessCell(
                    perturbation=spec.name,
                    level=level,
                    n=report.n_samples,
                    accuracy=report.accuracy,
                    f1_macro=report.f1_macro,
                )
            )
        curves.append(
            RobustnessCurve(perturbation=spec.name, clean_f1=baseline.f1_macro, cells=tuple(cells))
        )

    _logger.info(
        "robustness %r: %d perturbation(s) over %d test samples",
        config.model.name,
        len(curves),
        baseline.n,
    )
    return RobustnessReport(
        model=config.model.name,
        target=config.target.value,
        baseline=baseline,
        curves=tuple(curves),
    )


def _train_test(
    config: ExperimentConfig, samples: Sequence[Sample] | None
) -> tuple[list[Sample], list[Sample]]:
    """Return ``(train, test)`` splits, building them the same way an experiment does."""
    if samples is None:
        from tulip.data.builder import DatasetBuilder

        splits = DatasetBuilder(config.data).build(config.split, target=config.target)
    else:
        from tulip.data.splitting import speaker_disjoint_split

        splits = speaker_disjoint_split(samples, config.split)
    return splits.train, splits.test
