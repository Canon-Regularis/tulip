"""Tests for the dialect-evidence explainer (named-phenomenon attribution)."""

from __future__ import annotations

import pytest

from conftest import make_samples
from tulip.explain import EXPLAINERS, get_explainer
from tulip.explain.dialect_evidence import DialectEvidenceExplainer
from tulip.pipeline import DialectClassifier


@pytest.fixture(scope="module")
def fitted_pipeline():
    classifier = DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42)
    classifier.fit(make_samples())
    return classifier.pipeline_


def test_registered_under_canonical_name() -> None:
    assert "dialect_evidence" in EXPLAINERS.names()
    assert isinstance(get_explainer("dialect_evidence"), DialectEvidenceExplainer)


class TestEvidence:
    def test_marker_is_attributed_by_name(self, fitted_pipeline) -> None:
        explanation = DialectEvidenceExplainer().explain(fitted_pipeline, "baca poszedł na hale")
        tokens = [a.token for a in explanation.attributions]
        assert any("baca" in token for token in tokens)
        markers = explanation.details["markers"]
        assert any(m["label"] == "podhale" and m["surface"] == "baca" for m in markers)

    def test_fired_isogloss_is_attributed(self, fitted_pipeline) -> None:
        explanation = DialectEvidenceExplainer().explain(fitted_pipeline, "przyniósł psiwo")
        fired = explanation.details["fired_rules"]
        assert any(
            f["phenomenon"] == "isogloss_fired" and f["label"] == "soft_labials" for f in fired
        )
        # Masovian family is implicated by the fired soft labials.
        assert "masovian" in explanation.details["families"]

    def test_merger_environment_is_reported_not_fired(self, fitted_pipeline) -> None:
        explanation = DialectEvidenceExplainer().explain(fitted_pipeline, "jeszcze czekali")
        # mazurzenie is a merger: its standard environment is reported, but it
        # never appears among fired reflexes (it cannot be positively detected).
        applicable = {a["label"] for a in explanation.details["applicable_rules"]}
        fired = {f["label"] for f in explanation.details["fired_rules"]}
        assert "mazurzenie" in applicable
        assert "mazurzenie" not in fired

    def test_reports_the_models_prediction_and_caveat(self, fitted_pipeline) -> None:
        explanation = DialectEvidenceExplainer().explain(fitted_pipeline, "baca na hale")
        assert explanation.predicted_label is not None
        assert "resource-defined" in explanation.details["caveat"]

    def test_fired_evidence_outranks_a_single_marker(self, fitted_pipeline) -> None:
        # A fired sound change is weighted above a lexeme marker, so the fired
        # soft-labial cluster ranks above the 'baca' marker.
        explanation = DialectEvidenceExplainer().explain(fitted_pipeline, "baca psiwo")
        top = explanation.attributions[0].token
        assert "soft_labials" in top

    def test_standard_text_yields_no_positive_evidence(self, fitted_pipeline) -> None:
        explanation = DialectEvidenceExplainer().explain(
            fitted_pipeline, "wczoraj poszedł do sklepu"
        )
        assert not explanation.details["markers"]
        assert not explanation.details["fired_rules"]

    def test_rejects_empty_input(self, fitted_pipeline) -> None:
        from tulip.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            DialectEvidenceExplainer().explain(fitted_pipeline, "   ")
