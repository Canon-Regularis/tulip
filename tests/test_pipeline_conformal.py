"""Tests for split conformal prediction (tulip.pipeline.conformal)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_samples
from tulip.core.exceptions import ConfigurationError, DataError
from tulip.pipeline import ConformalClassifier, DialectClassifier
from tulip.pipeline.conformal import _conformal_quantile


@pytest.fixture(scope="module")
def base_and_calibration():
    """A fitted base classifier plus a held-out calibration split."""
    samples = make_samples(repeats=6)
    order = np.random.default_rng(0).permutation(len(samples))
    shuffled = [samples[i] for i in order]
    cut = int(0.7 * len(shuffled))
    train, calibration = shuffled[:cut], shuffled[cut:]
    base = DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42)
    base.fit(train)
    return base, calibration


class TestQuantile:
    def test_finite_sample_rank(self) -> None:
        # scores sorted [0.1,0.2,0.3,0.4]; alpha=0.25 -> rank=ceil(5*0.75)=4 -> 0.4
        assert _conformal_quantile(np.array([0.3, 0.1, 0.4, 0.2]), 0.25) == pytest.approx(0.4)

    def test_too_few_points_returns_full_coverage(self) -> None:
        # n=1, alpha=0.1 -> rank=ceil(2*0.9)=2 > 1 -> 1.0 (include every class)
        assert _conformal_quantile(np.array([0.3]), 0.1) == 1.0

    def test_empty_returns_full_coverage(self) -> None:
        assert _conformal_quantile(np.array([]), 0.1) == 1.0


class TestPredictionSets:
    def test_sets_are_non_empty_and_contain_the_top_label(self, base_and_calibration) -> None:
        base, calibration = base_and_calibration
        conformal = ConformalClassifier(base, alpha=0.1).fit_conformal(calibration)
        for prediction in conformal.predict_set([s.text for s in calibration]):
            assert 1 <= prediction.set_size <= len(base.classes_)
            assert prediction.contains(prediction.top_label)

    def test_lower_alpha_gives_larger_sets(self, base_and_calibration) -> None:
        base, calibration = base_and_calibration
        texts = [s.text for s in calibration]
        loose = ConformalClassifier(base, alpha=0.4).fit_conformal(calibration)
        strict = ConformalClassifier(base, alpha=0.05).fit_conformal(calibration)
        loose_size = np.mean([p.set_size for p in loose.predict_set(texts)])
        strict_size = np.mean([p.set_size for p in strict.predict_set(texts)])
        assert strict_size >= loose_size

    def test_is_deterministic(self, base_and_calibration) -> None:
        base, calibration = base_and_calibration
        texts = [s.text for s in calibration]
        a = ConformalClassifier(base, alpha=0.1).fit_conformal(calibration).predict_set(texts)
        b = ConformalClassifier(base, alpha=0.1).fit_conformal(calibration).predict_set(texts)
        assert [p.prediction_set for p in a] == [p.prediction_set for p in b]


class TestCoverage:
    def test_marginal_coverage_meets_target(self, base_and_calibration) -> None:
        base, calibration = base_and_calibration
        # By construction, coverage measured on the calibration set is at least
        # the 1 - alpha target (the threshold is the quantile of those scores).
        conformal = ConformalClassifier(base, alpha=0.2).fit_conformal(calibration)
        report = conformal.evaluate_coverage(calibration)
        assert report.coverage >= report.target_coverage
        assert report.mean_set_size <= len(base.classes_)

    def test_mondrian_is_class_conditional(self, base_and_calibration) -> None:
        base, calibration = base_and_calibration
        conformal = ConformalClassifier(base, alpha=0.2, mondrian=True).fit_conformal(calibration)
        # A per-class threshold exists for every class.
        assert set(conformal.qhat_) == set(base.classes_)
        report = conformal.evaluate_coverage(calibration)
        assert report.mondrian and report.coverage >= 0.5


class TestValidation:
    def test_alpha_out_of_range_raises(self, base_and_calibration) -> None:
        base, _ = base_and_calibration
        with pytest.raises(ConfigurationError, match="alpha"):
            ConformalClassifier(base, alpha=1.5)

    def test_predict_before_fit_raises(self, base_and_calibration) -> None:
        base, _ = base_and_calibration
        with pytest.raises(ConfigurationError, match="fit_conformal"):
            ConformalClassifier(base, alpha=0.1).predict_set(["some text"])

    def test_calibration_without_known_labels_raises(self, base_and_calibration) -> None:
        from tulip.core.types import DialectLabels, Sample

        base, _ = base_and_calibration
        alien = [
            Sample(id="x", text="tekst", speaker_id="s", labels=DialectLabels(dialect="atlantis"))
        ]
        with pytest.raises(DataError, match="known to the base"):
            ConformalClassifier(base, alpha=0.1).fit_conformal(alien)
