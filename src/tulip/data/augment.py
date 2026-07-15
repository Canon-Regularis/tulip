"""Linguistically-grounded training augmentation.

Grows a training set with perturbed copies of its text samples, drawn from the
same seeded engine the robustness sweep uses. Because the perturbations are
grounded in the Polish phonological rules, an augmented copy is a plausible move
along the standard-to-dialect axis, not generic noise.

The perturbation engine is imported lazily inside :func:`augment_samples`, so
``import tulip.data`` stays free of the feature and model stacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.core.types import Sample
    from tulip.robustness.report import AugmentSpec

__all__ = ["augment_samples"]


def augment_samples(samples: Sequence[Sample], spec: AugmentSpec) -> list[Sample]:
    """Return the originals plus ``spec.multiplier`` perturbed copies per text sample.

    For each round and each text sample, one perturbation and one of its levels
    are drawn from a single seeded stream, then applied. Audio-only samples are
    copied through unchanged. The result is deterministic given ``spec.seed``:
    same seed yields identical texts, ids, and set size.

    Args:
        samples: The training samples to augment.
        spec: Which perturbations to draw from, and how many copies to add.

    Returns:
        The original samples followed by the augmented copies.
    """
    from tulip.robustness import PERTURBATIONS

    originals = list(samples)
    if not spec.perturbations:
        return originals

    rng = np.random.default_rng(spec.seed)
    built = [PERTURBATIONS.create(entry.name, **entry.params) for entry in spec.perturbations]
    augmented = list(originals)
    for round_index in range(spec.multiplier):
        for sample in samples:
            if sample.text is None:
                continue
            choice = int(rng.integers(len(spec.perturbations)))
            entry = spec.perturbations[choice]
            level = entry.levels[int(rng.integers(len(entry.levels)))]
            new_text = built[choice].perturb(sample.text, level=level, rng=rng)
            augmented.append(
                sample.model_copy(update={"id": f"{sample.id}#aug{round_index}", "text": new_text})
            )
    return augmented
