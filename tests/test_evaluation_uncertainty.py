"""Tests for aleatoric vs epistemic uncertainty decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_samples
from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.uncertainty import (
    decompose_uncertainty,
    member_probabilities,
    uncertainty_report,
)
from tulip.pipeline import DialectClassifier

_VOTING = {"name": "voting", "params": {"estimators": ["logistic_regression", "naive_bayes"]}}


def _voting_classifier() -> DialectClassifier:
    return DialectClassifier(model=_VOTING, features=["char_tfidf"], seed=0).fit(
        make_samples(repeats=3)
    )


# --------------------------------------------------------- pure decomposition


def test_agreement_has_no_epistemic_uncertainty() -> None:
    identical = np.stack([np.array([[0.7, 0.2, 0.1]])] * 3)
    total, aleatoric, epistemic = decompose_uncertainty(identical)
    assert epistemic[0] == pytest.approx(0.0, abs=1e-9)
    assert total[0] == pytest.approx(aleatoric[0])


def test_disagreement_raises_epistemic_uncertainty() -> None:
    confident_agree = np.stack([np.array([[0.98, 0.01, 0.01]])] * 3)
    disagree = np.stack(
        [
            np.array([[0.9, 0.05, 0.05]]),
            np.array([[0.05, 0.9, 0.05]]),
            np.array([[0.05, 0.05, 0.9]]),
        ]
    )
    _, _, epistemic_agree = decompose_uncertainty(confident_agree)
    total, aleatoric, epistemic = decompose_uncertainty(disagree)
    assert epistemic[0] > epistemic_agree[0]
    assert total[0] >= aleatoric[0] - 1e-9  # Jensen: total >= aleatoric


def test_decompose_rejects_bad_input() -> None:
    with pytest.raises(ConfigurationError, match="n_members"):
        decompose_uncertainty(np.zeros((2, 3)))  # not 3-D
    with pytest.raises(ConfigurationError, match="two members"):
        decompose_uncertainty(np.zeros((1, 2, 3)))  # one member


# ------------------------------------------------------------- extraction


def test_member_probabilities_shape_and_normalisation() -> None:
    classifier = _voting_classifier()
    texts = [sample.text or "" for sample in make_samples(repeats=1)][:5]
    members = member_probabilities(classifier, texts)
    assert members.shape == (2, len(texts), len(classifier.classes_))
    assert np.allclose(members.sum(axis=2), 1.0)  # each member row is a distribution


def test_member_probabilities_rejects_non_ensemble() -> None:
    classifier = DialectClassifier(
        model="logistic_regression", features=["char_tfidf"], seed=0
    ).fit(make_samples(repeats=3))
    with pytest.raises(ConfigurationError, match="ensemble"):
        member_probabilities(classifier, ["jakiś tekst"])


def test_uncertainty_report_orders_and_renders() -> None:
    classifier = _voting_classifier()
    report = uncertainty_report(classifier, make_samples(repeats=1))
    assert report.n_members == 2
    assert report.mean_total >= report.mean_aleatoric - 1e-9
    assert report.mean_epistemic >= 0.0
    assert "Uncertainty" in report.to_markdown()
