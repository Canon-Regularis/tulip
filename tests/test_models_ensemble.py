"""Tests for the voting and stacking ensembles (tulip.models.ensemble)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_samples
from tulip.core.exceptions import ConfigurationError
from tulip.models import MODELS
from tulip.models.ensemble import _build_bases, _parse_entry
from tulip.pipeline import DialectClassifier


class TestRegistration:
    def test_voting_and_stacking_are_registered(self) -> None:
        assert "voting" in MODELS.names()
        assert "stacking" in MODELS.names()


class TestEndToEnd:
    @pytest.mark.parametrize("ensemble", ["voting", "stacking"])
    def test_trains_and_predicts_in_a_pipeline(self, ensemble: str) -> None:
        classifier = DialectClassifier(
            model={
                "name": ensemble,
                "params": {"estimators": ["naive_bayes", "logistic_regression"]},
            },
            features=["char_tfidf"],
            seed=42,
        )
        classifier.fit(make_samples())
        prediction = classifier.predict("Hej baca się pyto kaj się owce pasą na holi.")
        assert prediction.label in classifier.classes_
        # A full probability distribution over the known classes.
        assert prediction.probabilities
        assert sum(cp.probability for cp in prediction.probabilities) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_is_deterministic(self) -> None:
        def _fit_predict() -> str | None:
            clf = DialectClassifier(
                model={
                    "name": "stacking",
                    "params": {"estimators": ["logistic_regression", "random_forest"]},
                },
                features=["char_tfidf"],
                seed=7,
            )
            clf.fit(make_samples())
            return clf.predict("U nos w boru psiwo warzą jesce po staremu.").label

        assert _fit_predict() == _fit_predict()


class TestBaseConstruction:
    def test_string_entries_become_named_pairs(self) -> None:
        pairs = _build_bases(["naive_bayes", "logistic_regression"], seed=0)
        assert [alias for alias, _ in pairs] == ["naive_bayes", "logistic_regression"]

    def test_duplicate_names_get_unique_aliases(self) -> None:
        pairs = _build_bases(["logistic_regression", "logistic_regression"], seed=0)
        assert [alias for alias, _ in pairs] == ["logistic_regression", "logistic_regression_2"]

    def test_mapping_entry_with_alias_and_params(self) -> None:
        name, alias, params = _parse_entry(
            {"name": "logistic_regression", "alias": "lr", "params": {"C": 0.5}}
        )
        assert (name, alias, params) == ("logistic_regression", "lr", {"C": 0.5})

    def test_bad_entry_type_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="name or a mapping"):
            _parse_entry(123)


class TestValidation:
    def test_empty_estimators_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one"):
            MODELS.create("voting", estimators=[])

    def test_unknown_voting_mode_raises(self) -> None:
        with pytest.raises(ConfigurationError, match=r"soft.*hard"):
            MODELS.create("voting", estimators=["naive_bayes"], voting="fuzzy")

    def test_weight_length_mismatch_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="weights"):
            MODELS.create(
                "voting", estimators=["naive_bayes", "logistic_regression"], weights=[1.0]
            )

    def test_stacking_cv_floor(self) -> None:
        with pytest.raises(ConfigurationError, match="cv must be"):
            MODELS.create("stacking", estimators=["naive_bayes"], cv=1)

    def test_unexpected_params_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="unexpected"):
            MODELS.create("voting", estimators=["naive_bayes"], bogus=1)


class TestRawEstimator:
    def test_soft_voting_averages_probabilities(self) -> None:
        # Two identical bases -> the averaged proba equals a single base's proba.
        model = MODELS.create("voting", estimators=["logistic_regression"], voting="soft")
        x = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]])
        y = ["a", "b", "a", "b"]
        model.fit(x, y)
        proba = model.predict_proba(x)
        assert proba.shape == (4, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        assert list(model.classes_) == ["a", "b"]
