"""The chain-rule projection and backoff walk for hierarchical classification.

:mod:`tulip.pipeline.hierarchical.classifier` owns the per-level classifiers and
the fit/predict/persist lifecycle. The math it delegates to lives here: the walk
from a fine prediction to the coarsest one a policy accepts, and the chain-rule
projection that restricts a dialect distribution to its predicted family. These
are pure functions of a sample's per-level predictions and the level layout, so
extracting them keeps the classifier focused on composition and lets the
projection be tested without training anything, mirroring the already-extracted
:mod:`tulip.pipeline.hierarchical.policies`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.core.types import ClassProbability, Prediction
from tulip.labels.taxonomy import LabelLevel, family_for
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tulip.pipeline.hierarchical.policies import BackoffPolicy

__all__ = ["project_onto_family", "resolve_prediction"]

_logger = get_logger(__name__)


def resolve_prediction(
    index: int,
    by_level: Mapping[LabelLevel, list[Prediction]],
    *,
    fine_to_coarse: Sequence[LabelLevel],
    coarser_of: Mapping[LabelLevel, LabelLevel],
    mask_to_coarse: bool,
    policy: BackoffPolicy,
) -> Prediction:
    """Walk fine -> coarse for one sample, returning the accepted level.

    The walk starts at the finest level; if the policy accepts that (optionally
    coarse-masked) prediction it is returned, otherwise it steps one level
    coarser and retries. The coarsest level is always returned when nothing finer
    is accepted.
    """
    *backoff_levels, coarsest = fine_to_coarse
    for level in backoff_levels:
        candidate = _candidate(
            level, index, by_level, coarser_of=coarser_of, mask_to_coarse=mask_to_coarse
        )
        # ``None`` means this level cannot express the coarser decision at all
        # (e.g. the family is ``standard``, which has no dialects). That is not
        # a low-confidence answer; it is no answer, so back off immediately.
        if candidate is not None and policy.accepts(candidate):
            return candidate
    coarsest_candidate = _candidate(
        coarsest, index, by_level, coarser_of=coarser_of, mask_to_coarse=mask_to_coarse
    )
    assert coarsest_candidate is not None  # noqa: S101  # the coarsest level never projects
    return coarsest_candidate


def _candidate(
    level: LabelLevel,
    index: int,
    by_level: Mapping[LabelLevel, list[Prediction]],
    *,
    coarser_of: Mapping[LabelLevel, LabelLevel],
    mask_to_coarse: bool,
) -> Prediction | None:
    """Return the prediction at ``level``, projected onto its coarser neighbour.

    ``None`` signals that ``level`` cannot represent the coarser prediction.
    """
    prediction = by_level[level][index]
    coarser = coarser_of.get(level)
    if mask_to_coarse and level is LabelLevel.DIALECT and coarser is LabelLevel.FAMILY:
        return project_onto_family(prediction, by_level[coarser][index])
    return prediction


def project_onto_family(fine: Prediction, coarse: Prediction) -> Prediction | None:
    """Restrict the dialect distribution to the predicted family, by the chain rule.

    The dialect classes outside the coarse prediction's family drop to zero, and
    the survivors are rescaled to ``P(family) * P(dialect | family)``.

    Rescaling to the *coarse probability* rather than renormalising to 1.0 is the
    whole point. A family with exactly one dialect (Kashubian -> Kashubia) would
    otherwise renormalise to a certainty of 1.000 no matter how unsure the family
    classifier was, which silently defeats every confidence-based backoff policy.
    Under the chain rule that same case yields exactly ``P(family)``, so the fine
    prediction can never be more confident than the coarse decision it rests on.
    The returned distribution therefore sums to ``P(family)``, not to 1; it is a
    joint, not a conditional.

    Returns:
        The projected prediction, or ``None`` when the predicted family has no
        dialect children (``standard``), i.e. the finer level cannot answer.
    """
    if coarse.label is None:
        return fine  # the family classifier abstained; nothing to project onto
    consistent = {
        cp.label: cp.probability
        for cp in fine.probabilities
        if (family := family_for(cp.label)) is not None and family.value == coarse.label
    }
    total = sum(consistent.values())
    if total <= 0.0:
        _logger.debug(
            "family %r has no dialect children; the dialect level cannot answer",
            coarse.label,
        )
        return None
    family_probability = coarse.as_dict().get(coarse.label, 0.0)
    projected = tuple(
        ClassProbability(
            label=cp.label,
            probability=(
                family_probability * consistent[cp.label] / total
                if cp.label in consistent
                else 0.0
            ),
        )
        for cp in fine.probabilities
    )
    top_label = max(consistent, key=lambda label: consistent[label])
    return Prediction(
        label=top_label,
        level=fine.level,
        probabilities=projected,
        abstained=fine.abstained,
    )
