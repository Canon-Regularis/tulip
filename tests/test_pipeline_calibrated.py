"""Probability calibrators and calibrated abstention (no optional deps).

Deliberately self-contained: it does not import ``tulip.evaluation.calibration``
(another agent owns that). Negative log-likelihood is recomputed inline so the
"calibration lowers NLL" claim is checked against a local definition, not a
tulip helper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from conftest import make_samples
from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import DialectLabels, Prediction, Sample, TaskType
from tulip.labels.taxonomy import LabelLevel
from tulip.models.calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    ProbabilityCalibrator,
    TemperatureScaling,
)
from tulip.pipeline import DialectClassifier, evaluate_samples
from tulip.pipeline.calibrated import CalibratedClassifier
from tulip.pipeline.protocols import SamplePredictor

#: The three calibrator implementations, exercised against the shared contract.
CALIBRATOR_FACTORIES = [IdentityCalibrator, TemperatureScaling, IsotonicCalibrator]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(proba: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true classes (defined inline)."""
    picked = proba[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


def _confident_data(
    seed: int, *, sharpen: float, n: int = 800, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from a true distribution; probabilities mis-scaled by ``sharpen``.

    ``sharpen < 1`` sharpens the model's probabilities relative to reality
    (over-confidence), so a fitted temperature should be ``> 1``; ``sharpen > 1``
    flattens them (under-confidence), so the temperature should be ``< 1``.
    """
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(scale=2.0, size=(n, k))
    true_p = _softmax(true_logits)
    y = np.array([rng.choice(k, p=true_p[i]) for i in range(n)])
    model_p = _softmax(true_logits / sharpen)
    return model_p, y


class _ConstProbaBase:
    """A ``DialectClassifier``-shaped stand-in returning fixed probabilities.

    Implements only the surface ``CalibratedClassifier`` touches, so tests can
    pin the exact top probability instead of coaxing it out of a trained model.
    """

    task = TaskType.TEXT

    def __init__(
        self, classes: tuple[str, ...], row: object, target: LabelLevel = LabelLevel.DIALECT
    ) -> None:
        self.classes_ = tuple(classes)
        self._row = np.asarray(row, dtype=float)
        self.target = target

    def predict_proba(self, raws: object) -> np.ndarray:
        return np.tile(self._row, (len(list(raws)), 1))  # type: ignore[arg-type]


@pytest.fixture
def calibrated_text() -> tuple[CalibratedClassifier, list[Sample], DialectClassifier]:
    """A real text base + a temperature-calibrated wrapper on a disjoint split.

    Speaker ``spk3`` is held out for calibration/evaluation, so the calibrator
    never sees the probabilities the base was trained on.
    """
    samples = make_samples(repeats=4)
    train = [s for s in samples if not (s.speaker_id or "").endswith("spk3")]
    holdout = [s for s in samples if (s.speaker_id or "").endswith("spk3")]
    base = DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42)
    base.fit(train)
    wrapped = CalibratedClassifier(base, TemperatureScaling())
    wrapped.fit_calibration(holdout)
    return wrapped, holdout, base


# --------------------------------------------------------------------------- #
# Temperature scaling direction and NLL                                        #
# --------------------------------------------------------------------------- #


def test_overconfidence_yields_temperature_above_one_and_lowers_nll() -> None:
    train_p, train_y = _confident_data(0, sharpen=0.5)
    test_p, test_y = _confident_data(1, sharpen=0.5)
    calibrator = TemperatureScaling().fit(train_p, train_y)

    assert calibrator.temperature_ > 1.0
    assert _nll(calibrator.transform(test_p), test_y) < _nll(test_p, test_y)


def test_underconfidence_yields_temperature_below_one() -> None:
    train_p, train_y = _confident_data(2, sharpen=2.0)
    assert TemperatureScaling().fit(train_p, train_y).temperature_ < 1.0


def test_temperature_fit_is_deterministic() -> None:
    train_p, train_y = _confident_data(3, sharpen=0.5)
    first = TemperatureScaling().fit(train_p, train_y).temperature_
    second = TemperatureScaling().fit(train_p, train_y).temperature_
    assert first == second


# --------------------------------------------------------------------------- #
# Shared calibrator contract, checked for every implementation                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("factory", CALIBRATOR_FACTORIES)
def test_calibrator_preserves_shape_and_normalises(
    factory: type[ProbabilityCalibrator],
) -> None:
    train_p, train_y = _confident_data(4, sharpen=0.5)
    test_p, _ = _confident_data(5, sharpen=0.5)
    out = factory().fit(train_p, train_y).transform(test_p)

    assert out.shape == test_p.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("factory", CALIBRATOR_FACTORIES)
def test_transform_before_fit_raises_not_fitted(factory: type[ProbabilityCalibrator]) -> None:
    with pytest.raises(NotFittedError):
        factory().transform(np.array([[0.5, 0.5]]))


@pytest.mark.parametrize("factory", CALIBRATOR_FACTORIES)
def test_transform_column_mismatch_raises_configuration_error(
    factory: type[ProbabilityCalibrator],
) -> None:
    train_p, train_y = _confident_data(6, sharpen=0.5)  # three classes
    calibrator = factory().fit(train_p, train_y)
    with pytest.raises(ConfigurationError):
        calibrator.transform(np.array([[0.25, 0.25, 0.25, 0.25]]))


@pytest.mark.parametrize("factory", CALIBRATOR_FACTORIES)
def test_calibrators_satisfy_the_protocol(factory: type[ProbabilityCalibrator]) -> None:
    assert isinstance(factory(), ProbabilityCalibrator)


def test_identity_returns_bit_identical_probabilities() -> None:
    proba, y = _confident_data(7, sharpen=0.5)
    out = IdentityCalibrator().fit(proba, y).transform(proba)
    assert np.array_equal(out, proba)


def test_isotonic_zero_row_becomes_uniform() -> None:
    # Two cleanly separated classes: both regressors map the low input 0.1 to 0,
    # so a row of two low values collapses to [0, 0] and must be rescued to the
    # uniform distribution rather than emitting NaN.
    proba = np.array([[0.9, 0.1]] * 5 + [[0.1, 0.9]] * 5)
    y = np.array([0] * 5 + [1] * 5)
    out = IsotonicCalibrator().fit(proba, y).transform(np.array([[0.1, 0.1]]))

    assert np.isfinite(out).all()
    assert np.allclose(out, [[0.5, 0.5]])


# --------------------------------------------------------------------------- #
# CalibratedClassifier                                                         #
# --------------------------------------------------------------------------- #


def test_calibrated_abstention_softens_top_below_threshold() -> None:
    train_p, train_y = _confident_data(8, sharpen=0.5)
    temperature = TemperatureScaling().fit(train_p, train_y)
    assert temperature.temperature_ > 1.0

    raw_row = np.array([0.82, 0.10, 0.08])
    raw_top = float(raw_row.max())
    calibrated_top = float(temperature.transform(raw_row[None, :]).max())
    assert calibrated_top < raw_top  # softening lowered the peak
    threshold = (calibrated_top + raw_top) / 2.0

    base = _ConstProbaBase(classes=("podhale", "silesia", "kurpie"), row=raw_row)
    wrapped = CalibratedClassifier(base, temperature, abstain_threshold=threshold)  # type: ignore[arg-type]
    prediction = wrapped.predict_batch(["dowolny tekst"])[0]
    assert prediction.abstained
    assert prediction.label is None

    # Control: the identical base without softening keeps the top above the
    # threshold, so the abstention above is caused by calibration, not the base.
    identity = IdentityCalibrator().fit(raw_row[None, :], np.array([0]))
    control = CalibratedClassifier(base, identity, abstain_threshold=threshold)  # type: ignore[arg-type]
    kept = control.predict_batch(["dowolny tekst"])[0]
    assert not kept.abstained
    assert kept.label == "podhale"


def test_none_calibrator_installs_identity_null_object() -> None:
    base = _ConstProbaBase(classes=("a", "b"), row=[0.7, 0.3])
    assert isinstance(CalibratedClassifier(base).calibrator, IdentityCalibrator)  # type: ignore[arg-type]


def test_out_of_range_threshold_raises_configuration_error() -> None:
    base = _ConstProbaBase(classes=("a", "b"), row=[0.7, 0.3])
    with pytest.raises(ConfigurationError):
        CalibratedClassifier(base, abstain_threshold=1.5)  # type: ignore[arg-type]


def test_wrapped_is_sample_predictor_but_not_a_classifier(
    calibrated_text: tuple[CalibratedClassifier, list[Sample], DialectClassifier],
) -> None:
    wrapped, _, _ = calibrated_text
    assert isinstance(wrapped, SamplePredictor)
    assert not isinstance(wrapped, DialectClassifier)


def test_evaluate_samples_runs_via_delegated_attributes(
    calibrated_text: tuple[CalibratedClassifier, list[Sample], DialectClassifier],
) -> None:
    wrapped, holdout, base = calibrated_text
    report = evaluate_samples(wrapped, holdout)

    assert 0.0 <= report.accuracy <= 1.0
    # Temperature scaling is monotone within a row, so the arg-max (and hence
    # accuracy) is identical to the uncalibrated base classifier.
    assert report.accuracy == evaluate_samples(base, holdout).accuracy


def test_predict_samples_returns_one_prediction_per_sample(
    calibrated_text: tuple[CalibratedClassifier, list[Sample], DialectClassifier],
) -> None:
    wrapped, holdout, _ = calibrated_text
    predictions = wrapped.predict_samples(holdout)

    assert len(predictions) == len(holdout)
    assert all(isinstance(p, Prediction) for p in predictions)
    assert all(p.level is LabelLevel.DIALECT for p in predictions)


def test_predict_proba_is_calibrated_yet_preserves_the_hard_label(
    calibrated_text: tuple[CalibratedClassifier, list[Sample], DialectClassifier],
) -> None:
    wrapped, _, base = calibrated_text
    raws = ["Hej, baca się pyto, kaj się owce pasą na holi."]
    calibrated = wrapped.predict_proba(raws)

    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.isfinite(calibrated).all()
    assert np.argmax(calibrated, axis=1)[0] == np.argmax(base.predict_proba(raws), axis=1)[0]


def test_fit_calibration_rejects_empty_and_labelless_sets() -> None:
    base = DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42)
    base.fit(make_samples(repeats=3))
    wrapped = CalibratedClassifier(base, TemperatureScaling())

    with pytest.raises(DataError):
        wrapped.fit_calibration([])

    labelless = [Sample(id="x1", text="jakiś tekst bez etykiety", labels=DialectLabels())]
    with pytest.raises(DataError):
        wrapped.fit_calibration(labelless)


def test_predict_samples_missing_modality_raises_data_error() -> None:
    base = _ConstProbaBase(classes=("a", "b"), row=[0.7, 0.3])  # text task
    wrapped = CalibratedClassifier(base, IdentityCalibrator())  # type: ignore[arg-type]
    audio_only = Sample(id="a1", audio_path=Path("nagranie.wav"))
    with pytest.raises(DataError):
        wrapped.predict_samples([audio_only])
