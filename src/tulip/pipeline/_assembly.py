"""Shared building blocks for the classifier facade and its wrappers.

The plain :class:`~tulip.pipeline.classifier.DialectClassifier`, the
:class:`~tulip.pipeline.calibrated.CalibratedClassifier`, the
:class:`~tulip.pipeline.conformal.ConformalClassifier`, and the
:class:`~tulip.pipeline.fusion.MultimodalClassifier` share four small pieces of
logic. Keeping them here, in a module that imports only from ``core``, keeps the
copies provably identical and avoids reaching into each other's internals:

* :func:`predictions_from_proba` assembles ranked :class:`Prediction` records.
* :func:`raw_input_of` and :func:`raws_for_task` read a sample's modality input.
* :func:`align_in_vocab_rows` drops calibration rows whose label is out of vocab.
* :func:`validate_abstain_threshold` guards the shared ``abstain_threshold`` knob.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import ClassProbability, Prediction, TaskType

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from tulip.core.types import Sample
    from tulip.labels.taxonomy import LabelLevel

__all__ = [
    "align_in_vocab_rows",
    "predictions_from_proba",
    "raw_input_of",
    "raws_for_task",
    "validate_abstain_threshold",
]


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
    abstention, so no row is ever abstained; the behaviour the multimodal
    classifier relies on.

    Building rich, validated :class:`Prediction` objects costs ~40% of batch
    wall time on top of ``predict_proba`` (measured; pydantic's
    ``model_construct`` fast path bought nothing, so the validated constructor
    stays). Bulk consumers that only need the probability matrix (evaluation
    does) should call ``predict_proba`` directly and skip this assembly.

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


def raw_input_of(sample: Sample, task: TaskType) -> Any | None:
    """Return a sample's raw model input for ``task``, or ``None`` if absent."""
    return sample.text if task is TaskType.TEXT else sample.audio_path


def raws_for_task(samples: Sequence[Sample], task: TaskType) -> list[Any]:
    """Extract each sample's raw input for ``task``, erroring on a missing one.

    Raises:
        DataError: if any sample lacks the ``task`` modality; prediction has no
            label to fall back on, so a missing input is an error, not a skip.
    """
    raws = [raw_input_of(sample, task) for sample in samples]
    missing = [sample.id for sample, raw in zip(samples, raws, strict=True) if raw is None]
    if missing:
        raise DataError(
            f"{len(missing)} sample(s) carry no {task.value} input and cannot be "
            f"classified (first: {missing[0]!r})"
        )
    return raws


def align_in_vocab_rows(
    labels: Sequence[str], classes: Sequence[str]
) -> tuple[list[int], list[int]]:
    """Map labels onto class indices, dropping rows whose label is out of vocab.

    Args:
        labels: Gold labels, one per calibration row.
        classes: The classifier's class vocabulary.

    Returns:
        ``(kept_rows, y_index)``: the surviving row positions and their class
        indices, aligned. Callers decide whether an empty result is an error.
    """
    index_of = {label: index for index, label in enumerate(classes)}
    kept_rows: list[int] = []
    y_index: list[int] = []
    for row, label in enumerate(labels):
        class_index = index_of.get(label)
        if class_index is None:
            continue
        kept_rows.append(row)
        y_index.append(class_index)
    return kept_rows, y_index


def validate_abstain_threshold(value: float | None) -> None:
    """Raise if an abstain threshold is set but outside ``[0, 1]``.

    Raises:
        ConfigurationError: if ``value`` is not ``None`` and not in ``[0, 1]``.
    """
    if value is not None and not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"abstain_threshold must be within [0, 1], got {value}")
