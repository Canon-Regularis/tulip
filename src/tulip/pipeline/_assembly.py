"""Shared assembly of :class:`Prediction` records from a probability matrix.

Three sibling classifiers -- the plain
:class:`~tulip.pipeline.classifier.DialectClassifier`, the
:class:`~tulip.pipeline.calibrated.CalibratedClassifier`, and the
:class:`~tulip.pipeline.fusion.MultimodalClassifier` -- each turned a probability
matrix into ranked :class:`~tulip.core.types.Prediction` records with the *same*
loop. Three independent audits flagged the duplication. Promoting it to one
private helper keeps the three provably identical: any change to how a
prediction is ranked, argmax-labelled, or abstained now happens in exactly one
place instead of drifting between copies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tulip.core.types import ClassProbability, Prediction

if TYPE_CHECKING:
    from tulip.labels.taxonomy import LabelLevel

__all__ = ["predictions_from_proba"]


def predictions_from_proba(
    proba: np.ndarray,
    classes: tuple[str, ...],
    level: LabelLevel,
    *,
    abstain_threshold: float | None = None,
) -> list[Prediction]:
    """Assemble ranked :class:`Prediction` records from a probability matrix.

    For each row of ``proba`` this builds the full ranked
    :class:`~tulip.core.types.ClassProbability` tuple (aligned to ``classes``),
    labels the sample with the argmax class, and abstains (``label=None``,
    ``abstained=True``) when the top probability falls below
    ``abstain_threshold``. Passing ``abstain_threshold=None`` disables
    abstention, so no row is ever abstained -- the behaviour the multimodal
    classifier relies on.

    Building rich, validated :class:`Prediction` objects costs ~40% of batch
    wall time on top of ``predict_proba`` (measured; pydantic's
    ``model_construct`` fast path bought nothing, so the validated constructor
    stays). Bulk consumers that only need the probability matrix -- evaluation
    does -- should call ``predict_proba`` directly and skip this assembly.

    Args:
        proba: Probability matrix, one row per sample, columns aligned to
            ``classes``.
        classes: Class-label vocabulary aligned to ``proba``'s columns.
        level: Label granularity stamped onto every returned prediction.
        abstain_threshold: When set, a row whose top probability is below it
            abstains instead of guessing; ``None`` never abstains.
    """
    predictions: list[Prediction] = []
    for row in proba:
        ranked = tuple(
            ClassProbability(label=label, probability=float(p))
            for label, p in zip(classes, row, strict=True)
        )
        top = float(np.max(row))
        abstained = abstain_threshold is not None and top < abstain_threshold
        predictions.append(
            Prediction(
                label=None if abstained else classes[int(np.argmax(row))],
                level=level,
                probabilities=ranked,
                abstained=abstained,
            )
        )
    return predictions
