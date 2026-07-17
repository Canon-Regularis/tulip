"""Shared building blocks for the classifier facade and its wrappers.

The plain :class:`~tulip.pipeline.classifier.DialectClassifier` and the
uncertainty wrappers (:class:`~tulip.pipeline.calibrated.CalibratedClassifier`,
:class:`~tulip.pipeline.conformal.ConformalClassifier`,
:class:`~tulip.pipeline.openset.OpenSetClassifier`), plus the
:class:`~tulip.pipeline.fusion.MultimodalClassifier`, share small pieces of
logic. Keeping them here, in a module that imports only from ``core`` and the
shared logger, keeps the copies provably identical and avoids reaching into each
other's internals:

* :func:`predictions_from_proba` assembles ranked :class:`Prediction` records.
* :func:`raw_input_of`, :func:`raws_for_task`, and :func:`keep_with_raw` read a
  sample's modality input.
* :func:`align_in_vocab_rows` and :func:`scored_in_vocab_rows` drop calibration
  rows whose label is out of vocabulary.
* :func:`require_labelled_batch` guards an empty labelled batch;
  :func:`conformal_row_sets` yields the per-row conformal decision.
* :func:`validate_abstain_threshold` guards the shared ``abstain_threshold`` knob.
* :class:`_BaseDelegating` forwards ``classes_``, ``target``, and ``task`` to the
  base classifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import ClassProbability, Prediction, TaskType
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any

    from tulip.core.types import Sample
    from tulip.labels.taxonomy import LabelLevel
    from tulip.pipeline.classifier import LabelledBatch
    from tulip.pipeline.protocols import CalibratableClassifier

__all__ = [
    "align_in_vocab_rows",
    "conformal_row_sets",
    "keep_with_raw",
    "predictions_from_proba",
    "raw_input_of",
    "raws_for_task",
    "require_labelled_batch",
    "scored_in_vocab_rows",
    "validate_abstain_threshold",
]

_logger = get_logger(__name__)


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


def keep_with_raw(samples: Sequence[Sample], task: TaskType) -> list[tuple[Sample, Any]]:
    """Pair each sample carrying the ``task`` input with its raw, dropping the rest.

    Samples without a ``task`` input cannot be predicted or pseudo-labelled, so
    the callers that must skip them (distillation, self-training, acquisition)
    share this one filter instead of open-coding it.

    Returns:
        The surviving ``(sample, raw)`` pairs, in input order.
    """
    pairs: list[tuple[Sample, Any]] = []
    for sample in samples:
        raw = raw_input_of(sample, task)
        if raw is not None:
            pairs.append((sample, raw))
    return pairs


def require_labelled_batch(
    base: CalibratableClassifier, samples: Sequence[Sample], *, context: str
) -> LabelledBatch:
    """Build the base's labelled batch, erroring when nothing usable survives.

    The uncertainty wrappers all need a non-empty batch of target-level labelled
    samples before they can calibrate or evaluate, and each raised the same
    "no usable samples for target X (skipped N)" error. This centralises it.

    Args:
        base: A fitted classifier exposing ``labelled_batch`` and ``target``.
        samples: Candidate labelled samples.
        context: A short phrase naming what needs the data; it completes the
            error message after the skipped-count clause.

    Returns:
        The non-empty :class:`~tulip.pipeline.classifier.LabelledBatch`.

    Raises:
        DataError: when no sample survives with a target-level label.
    """
    batch = base.labelled_batch(samples)
    if not batch.raws:
        raise DataError(
            f"no usable samples for target {base.target.value!r} "
            f"(skipped {batch.n_skipped}); {context}"
        )
    return batch


def scored_in_vocab_rows(
    base: CalibratableClassifier,
    batch: LabelledBatch,
    *,
    log_label: str,
    empty_error: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a labelled batch and keep only its in-vocabulary rows.

    Runs ``predict_proba`` over the batch, maps gold labels to class indices via
    :func:`align_in_vocab_rows`, drops rows whose label the base never saw at
    training time (logging how many), and returns the surviving probabilities and
    their true-class indices. The conformal and calibration wrappers share this.

    Args:
        base: The fitted classifier providing ``predict_proba`` and ``classes_``.
        batch: A non-empty labelled batch (see :func:`require_labelled_batch`).
        log_label: Prefix for the dropped-row log line, e.g. ``"conformal"``.
        empty_error: Message for the :class:`DataError` raised when no row
            carries an in-vocabulary label.

    Returns:
        ``(proba_kept, y_index)``: the kept rows' probabilities and class indices.

    Raises:
        DataError: when no calibration row carries an in-vocabulary label.
    """
    proba = base.predict_proba(batch.raws)
    kept_rows, y_index = align_in_vocab_rows(batch.labels, base.classes_)
    if not kept_rows:
        raise DataError(empty_error)
    dropped = len(batch.labels) - len(kept_rows)
    if dropped:
        _logger.info(
            "%s: dropped %d/%d rows with labels unseen at training time",
            log_label,
            dropped,
            len(batch.labels),
        )
    return proba[kept_rows], np.asarray(y_index, dtype=int)


def conformal_row_sets(
    proba: np.ndarray, classes: Sequence[str], thresholds: np.ndarray
) -> Iterator[tuple[tuple[str, ...], int, np.ndarray]]:
    """Yield the conformal decision for each probability row.

    For each row this yields ``(included, top_index, row)`` where ``included`` is
    the labels whose nonconformity ``1 - p`` is at or below the per-class
    ``thresholds`` (most probable first) and ``top_index`` is the argmax column.
    The conformal and open-set wrappers build their different prediction objects
    from this one primitive rather than re-implementing the loop.
    """
    for row in proba:
        nonconformity = 1.0 - row
        included = tuple(
            classes[index]
            for index in np.argsort(row)[::-1]  # most probable first
            if nonconformity[index] <= thresholds[index]
        )
        yield included, int(np.argmax(row)), row


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


class _BaseDelegating:
    """Forward the read-only classifier facade to ``self.base``.

    The uncertainty wrappers each wrap a fitted base classifier and expose its
    class vocabulary, target level, and task unchanged, so
    :func:`~tulip.pipeline.experiment.evaluate_samples` treats a wrapped
    classifier exactly like a bare one. Inheriting this keeps those three
    delegators in one place. The subclass supplies ``base`` (an attribute or a
    property); this mixin only reads it.
    """

    base: CalibratableClassifier

    @property
    def classes_(self) -> tuple[str, ...]:
        """Class-label vocabulary, delegated to the base classifier."""
        return self.base.classes_

    @property
    def target(self) -> LabelLevel:
        """Target label granularity, delegated to the base classifier."""
        return self.base.target

    @property
    def task(self) -> TaskType:
        """Input modality, delegated to the base classifier."""
        return self.base.task
