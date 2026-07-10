"""A calibrating wrapper that makes ``abstain_threshold`` mean what it says.

:class:`~tulip.pipeline.classifier.DialectClassifier` abstains when a *raw* top
probability falls below its threshold. Raw neural/boosted probabilities are
systematically over-confident, so "abstain below 0.9" does not actually mean
"abstain when less than 90% likely to be right". :class:`CalibratedClassifier`
fixes this: it calibrates the base classifier's probabilities first (see
:mod:`tulip.models.calibration`) and only then thresholds them.

Why composition, not subclassing (LSP)
---------------------------------------
``CalibratedClassifier`` wraps a ``DialectClassifier`` by composition and is
**not** a subclass of it. Subclassing would advertise substitutability that does
not hold: a caller holding a ``DialectClassifier`` expects
:meth:`~tulip.pipeline.classifier.DialectClassifier.predict_proba` to return the
*model's own* probabilities, whereas this class returns *calibrated* ones -- a
silently different postcondition. Instead, ``CalibratedClassifier`` relates to
its base and to its siblings (hierarchical, multimodal) only through the narrow
:class:`~tulip.pipeline.protocols.SamplePredictor` protocol, and additionally
delegates the handful of attributes
(:attr:`classes_`, :attr:`target`, :attr:`task`, :meth:`labelled_batch`) that
:func:`~tulip.pipeline.experiment.evaluate_samples` reads, so a calibrated
classifier evaluates exactly like a bare one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import TaskType
from tulip.models.calibration import IdentityCalibrator
from tulip.pipeline._assembly import predictions_from_proba
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Self

    from tulip.core.types import Prediction, Sample
    from tulip.labels.taxonomy import LabelLevel
    from tulip.models.calibration import ProbabilityCalibrator
    from tulip.pipeline.classifier import DialectClassifier, LabelledBatch

__all__ = ["CalibratedClassifier"]

_logger = get_logger(__name__)


class CalibratedClassifier:
    """Calibrate a fitted classifier's probabilities, then abstain on them.

    Args:
        base: A fitted :class:`~tulip.pipeline.classifier.DialectClassifier`
            whose raw probabilities are to be calibrated.
        calibrator: How to calibrate; ``None`` installs an
            :class:`~tulip.models.calibration.IdentityCalibrator` (the Null
            Object), so probabilities pass through unchanged.
        abstain_threshold: When set, a prediction whose *calibrated* top
            probability falls below it abstains (``label=None``). Because the
            threshold is compared against a calibrated probability, it finally
            means "abstain when less than this likely to be right".

    Raises:
        ConfigurationError: if ``abstain_threshold`` is outside ``[0, 1]``.
    """

    def __init__(
        self,
        base: DialectClassifier,
        calibrator: ProbabilityCalibrator | None = None,
        *,
        abstain_threshold: float | None = None,
    ) -> None:
        if abstain_threshold is not None and not 0.0 <= abstain_threshold <= 1.0:
            raise ConfigurationError(
                f"abstain_threshold must be within [0, 1], got {abstain_threshold}"
            )
        self.base = base
        self.calibrator: ProbabilityCalibrator = (
            IdentityCalibrator() if calibrator is None else calibrator
        )
        self.abstain_threshold = abstain_threshold

    # ---------------------------------------------------------------- fit

    def fit_calibration(self, samples: Sequence[Sample]) -> Self:
        """Fit the calibrator on a HELD-OUT validation split.

        The samples MUST be validation data the base classifier never trained
        on. Fitting a calibrator on the very probabilities the base already fit
        to is the classic silent mistake: the model is over-confident *and*
        near-perfect there, so the fit learns "do nothing" and calibration
        buys nothing on unseen data. Pass a disjoint split.

        Rows whose gold label is unknown to the base classifier (labels it
        never saw at training time) are dropped with a logged count -- they
        carry no valid class index to calibrate against.

        Raises:
            DataError: if the calibration set yields no usable, in-vocabulary
                labelled samples.
        """
        batch = self.base.labelled_batch(samples)
        if not batch.raws:
            raise DataError(
                f"calibration set has no usable samples for target "
                f"{self.base.target.value!r} (skipped {batch.n_skipped}); "
                f"fit_calibration needs held-out, labelled validation data"
            )
        proba = self.base.predict_proba(batch.raws)
        index_of = {label: index for index, label in enumerate(self.base.classes_)}
        kept_rows: list[int] = []
        y_index: list[int] = []
        for row, label in enumerate(batch.labels):
            class_index = index_of.get(label)
            if class_index is None:
                continue
            kept_rows.append(row)
            y_index.append(class_index)
        if not kept_rows:
            raise DataError(
                "calibration set has no samples whose gold label is known to the base "
                "classifier; cannot fit a calibrator"
            )
        dropped = len(batch.labels) - len(kept_rows)
        if dropped:
            _logger.info(
                "calibration: dropped %d/%d rows with labels unseen at training time",
                dropped,
                len(batch.labels),
            )
        self.calibrator.fit(proba[kept_rows], np.asarray(y_index, dtype=int))
        return self

    # ------------------------------------------------------------ predict

    def predict_proba(self, raws: Sequence[Any]) -> np.ndarray:
        """Return the CALIBRATED probability matrix, columns aligned to ``classes_``.

        Raises:
            NotFittedError: if :meth:`fit_calibration` has not run (the
                calibrator refuses to transform before it is fitted).
        """
        return self.calibrator.transform(self.base.predict_proba(raws))

    def predict_batch(self, raws: Sequence[Any]) -> list[Prediction]:
        """Classify raw inputs, abstaining on the CALIBRATED top probability.

        This is the deliverable: :attr:`abstain_threshold` is compared against a
        calibrated probability, so the cutoff finally means what it says.
        """
        return predictions_from_proba(
            self.predict_proba(raws),
            self.base.classes_,
            self.base.target,
            abstain_threshold=self.abstain_threshold,
        )

    def predict_samples(self, samples: Sequence[Sample]) -> list[Prediction]:
        """Classify samples via the base's modality (satisfies SamplePredictor).

        Reads whichever modality the base was built for and applies calibrated
        abstention.

        Raises:
            DataError: if any sample lacks the base classifier's input modality.
        """
        return self.predict_batch(self._raws_of(samples))

    def _raws_of(self, samples: Sequence[Sample]) -> list[Any]:
        """Extract the base's raw model inputs, erroring on a missing modality.

        Mirrors the modality switch in :meth:`DialectClassifier.predict_samples`
        rather than reaching into the base's private ``_raw_of``.
        """
        task = self.base.task
        raws = [sample.text if task is TaskType.TEXT else sample.audio_path for sample in samples]
        missing = [sample.id for sample, raw in zip(samples, raws, strict=True) if raw is None]
        if missing:
            raise DataError(
                f"{len(missing)} sample(s) carry no {task.value} input and cannot be "
                f"classified (first: {missing[0]!r})"
            )
        return raws

    # ----------------------------------------------------------- delegates

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

    def labelled_batch(self, samples: Sequence[Sample]) -> LabelledBatch:
        """Pair raw inputs with target-level labels, delegated to the base.

        Present so :func:`~tulip.pipeline.experiment.evaluate_samples` treats a
        calibrated classifier exactly like a bare one.
        """
        return self.base.labelled_batch(samples)

    def __repr__(self) -> str:
        return (
            f"CalibratedClassifier(base={self.base!r}, "
            f"calibrator={type(self.calibrator).__name__}, "
            f"abstain_threshold={self.abstain_threshold})"
        )
