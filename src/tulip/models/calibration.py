"""Post-hoc probability calibrators for over- and under-confident classifiers.

Raw neural and boosted class probabilities are systematically miscalibrated: a
model that reports ``0.9`` is usually right far less than 90% of the time. Any
decision rule phrased in probability terms, above all
:attr:`~tulip.pipeline.classifier.DialectClassifier.abstain_threshold`, which is
meant to abstain "below 0.9", is meaningless until the probabilities are
mapped onto the scale they claim to live on. These calibrators perform that map.

Design
------
Every calibrator implements the narrow :class:`ProbabilityCalibrator` protocol
(ISP: two methods, nothing more) and is interchangeable behind it (DIP:
consumers depend on the protocol, never on a concrete class, so adding a new
calibrator never edits the code that uses one). :class:`IdentityCalibrator` is
the Null Object: a consumer configured with "no calibration" still holds a
real calibrator and never writes ``if calibrator is None``.

The shared contract, checked against *every* implementation in
``tests/test_pipeline_calibrated.py`` (this is the Liskov substitution principle
made concrete, no implementation may weaken it):

* the output has the same shape as the input;
* every output row is a probability distribution (sums to 1, no ``NaN``/``inf``);
* :class:`~sklearn.exceptions.NotFittedError` is raised if ``transform`` runs
  before ``fit`` (deliberately scikit-learn's error, not a :class:`TulipError`,
  so the calibrators read as ordinary estimators to sklearn-aware callers);
* :class:`~tulip.core.exceptions.ConfigurationError` is raised when the column
  count at ``transform`` differs from the one seen at ``fit``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression

from tulip.core.exceptions import ConfigurationError

__all__ = [
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "ProbabilityCalibrator",
    "TemperatureScaling",
]

#: Lower bound for clipping probabilities before taking a logarithm, so the
#: surrogate logits used by temperature scaling stay finite.
_LOG_CLIP_MIN = 1e-12

#: Interval scanned for the temperature scalar. Strictly above 0 (temperature
#: must be positive) and wide enough to cover both sharpening (``T < 1``) and
#: softening (``T > 1``) of any realistically miscalibrated model.
_TEMPERATURE_BOUNDS = (1e-2, 1e2)


@runtime_checkable
class ProbabilityCalibrator(Protocol):
    """Maps a classifier's probability matrix to a better-calibrated one.

    Columns keep their meaning (class ``j`` in stays class ``j`` out); only the
    numbers are re-scaled. Implementations honour the module-level contract.
    """

    def fit(self, proba: np.ndarray, y_index: np.ndarray) -> ProbabilityCalibrator:
        """Learn the calibration map from validation probabilities and labels.

        Args:
            proba: Probability matrix ``(n_samples, n_classes)``.
            y_index: True class *index* per row (a column into ``proba``), not a
                label string.
        """
        ...

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Return a calibrated copy of ``proba`` with the same shape."""
        ...


def _as_proba_matrix(proba: np.ndarray) -> np.ndarray:
    """Coerce ``proba`` to a 2-D float array, rejecting other ranks."""
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim != 2:
        raise ConfigurationError(
            f"probability matrix must be 2-D (n_samples, n_classes); got shape {arr.shape}"
        )
    return arr


def _encode_targets(y_index: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    """Validate and integer-cast the per-row true-class indices."""
    y = np.asarray(y_index)
    if y.ndim != 1 or y.shape[0] != n_rows:
        raise ConfigurationError(
            f"y_index must be a 1-D array of length {n_rows}; got shape {y.shape}"
        )
    encoded = y.astype(int)
    if n_rows and (encoded.min() < 0 or encoded.max() >= n_classes):
        raise ConfigurationError(
            f"y_index values must be class indices in [0, {n_classes}); "
            f"got range [{int(encoded.min())}, {int(encoded.max())}]"
        )
    return encoded


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _renormalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to sum to 1; an all-zero row becomes uniform (never NaN)."""
    row_sums = matrix.sum(axis=1)
    degenerate = row_sums <= 0.0
    safe_sums = np.where(degenerate, 1.0, row_sums)
    out = matrix / safe_sums[:, np.newaxis]
    if degenerate.any():
        out[degenerate, :] = 1.0 / matrix.shape[1]
    return out


class _FittableCalibrator:
    """Fitted-state bookkeeping and shape guards shared by the calibrators.

    Not public and not a calibrator itself: it only removes the boilerplate the
    shared contract (``NotFittedError`` before fit, column agreement between fit
    and transform) would otherwise repeat in every implementation.
    """

    #: Number of classes seen at fit time; ``None`` marks an unfitted calibrator.
    _n_classes: int | None = None

    def _remember(self, proba: np.ndarray) -> np.ndarray:
        """Validate a fit-time matrix and record its class count."""
        arr = _as_proba_matrix(proba)
        self._n_classes = arr.shape[1]
        return arr

    def _validated(self, proba: np.ndarray) -> np.ndarray:
        """Validate a transform-time matrix against the fitted class count."""
        if self._n_classes is None:
            raise NotFittedError(
                f"{type(self).__name__} is not fitted; call fit() before transform()"
            )
        arr = _as_proba_matrix(proba)
        if arr.shape[1] != self._n_classes:
            raise ConfigurationError(
                f"{type(self).__name__} was fitted on {self._n_classes} classes "
                f"but transform() received {arr.shape[1]}"
            )
        return arr


class IdentityCalibrator(_FittableCalibrator):
    """Null Object calibrator: returns probabilities untouched.

    Lets a consumer configured with "no calibration" still hold a real
    :class:`ProbabilityCalibrator`, so no code branches on ``None``. It still
    enforces the fit/transform contract, so substituting a real calibrator for
    it cannot change the surrounding control flow.
    """

    def fit(self, proba: np.ndarray, y_index: np.ndarray) -> IdentityCalibrator:
        """Record the class count; the map itself is the identity."""
        del y_index
        self._remember(proba)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Return the probabilities unchanged (bit-for-bit for a float array)."""
        return self._validated(proba)


class TemperatureScaling(_FittableCalibrator):
    """Single-parameter temperature scaling on probabilities.

    Temperature scaling divides the logits by a learned scalar ``T`` before the
    softmax: ``T > 1`` softens an over-confident distribution, ``T < 1`` sharpens
    an under-confident one, and the arg-max (hence the hard label) never moves.

    We are handed probabilities, not logits, so ``log(p)`` is used as surrogate
    logits. That is exact, not an approximation. With true logits ``z``,
    ``log(softmax(z)) = z - c`` for the per-row constant ``c = logsumexp(z)``,
    and softmax is invariant to a per-row additive constant, so
    ``softmax(log(p) / T) = softmax((z - c) / T) = softmax(z / T)``: textbook
    temperature scaling. ``p`` is clipped away from 0 before the logarithm.

    ``T`` is fitted by minimising the negative log-likelihood of the true
    classes over :data:`_TEMPERATURE_BOUNDS` with
    :func:`scipy.optimize.minimize_scalar`, a deterministic bounded scan.
    """

    #: Learned temperature, set by :meth:`fit`; ``> 0`` by construction.
    temperature_: float

    def fit(self, proba: np.ndarray, y_index: np.ndarray) -> TemperatureScaling:
        """Fit :attr:`temperature_` by minimising validation NLL."""
        arr = self._remember(proba)
        targets = _encode_targets(y_index, arr.shape[0], arr.shape[1])
        logits = np.log(np.clip(arr, _LOG_CLIP_MIN, 1.0))
        rows = np.arange(arr.shape[0])

        def negative_log_likelihood(temperature: float) -> float:
            picked = _softmax(logits / temperature)[rows, targets]
            return float(-np.mean(np.log(np.clip(picked, _LOG_CLIP_MIN, 1.0))))

        result = minimize_scalar(
            negative_log_likelihood, bounds=_TEMPERATURE_BOUNDS, method="bounded"
        )
        self.temperature_ = float(result.x)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Soften/sharpen the probabilities by the learned temperature."""
        arr = self._validated(proba)
        logits = np.log(np.clip(arr, _LOG_CLIP_MIN, 1.0))
        return _softmax(logits / self.temperature_)


class IsotonicCalibrator(_FittableCalibrator):
    """Non-parametric per-class isotonic calibration.

    Fits one monotonic one-vs-rest
    :class:`~sklearn.isotonic.IsotonicRegression` (``out_of_bounds="clip"``) per
    class, mapping that class's raw probability onto its empirical frequency,
    then renormalises each row back into a distribution. More flexible than
    temperature scaling: it can correct non-uniform miscalibration, at the
    cost of needing more data and being able to reorder classes. A row whose
    per-class outputs all fall to 0 is replaced by the uniform distribution, so
    the output never contains ``NaN``.
    """

    def fit(self, proba: np.ndarray, y_index: np.ndarray) -> IsotonicCalibrator:
        """Fit one isotonic regressor per class on one-vs-rest targets."""
        arr = self._remember(proba)
        targets = _encode_targets(y_index, arr.shape[0], arr.shape[1])
        self._regressors: list[IsotonicRegression] = []
        for class_index in range(arr.shape[1]):
            regressor = IsotonicRegression(out_of_bounds="clip")
            regressor.fit(arr[:, class_index], (targets == class_index).astype(np.float64))
            self._regressors.append(regressor)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Map each class column through its regressor, then renormalise rows."""
        arr = self._validated(proba)
        calibrated = np.empty_like(arr)
        for class_index, regressor in enumerate(self._regressors):
            calibrated[:, class_index] = regressor.predict(arr[:, class_index])
        return _renormalise(calibrated)
