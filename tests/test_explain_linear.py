"""Tests for the top_tfidf linear-coefficient explainer."""

from __future__ import annotations

import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from tulip.core.exceptions import ConfigurationError
from tulip.explain import get_explainer
from tulip.explain.linear import class_top_features

PODHALE_QUERY = "Hej, baca się pyto, kaj się owce pasą na holi."


@pytest.fixture
def fitted_pipeline(synthetic_texts_and_labels: tuple[list[str], list[str]]) -> Pipeline:
    texts, labels = synthetic_texts_and_labels
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=2000, random_state=0)),
        ]
    )
    return pipeline.fit(texts, labels)


def _class_vocabulary(texts: list[str], labels: list[str], pipeline: Pipeline) -> dict[str, set]:
    """Tokens per class, using the fitted vectorizer's own analyzer."""
    analyzer = pipeline.named_steps["tfidf"].build_analyzer()
    vocabulary: dict[str, set] = {}
    for text, label in zip(texts, labels, strict=True):
        vocabulary.setdefault(label, set()).update(analyzer(text))
    return vocabulary


def test_attributions_are_real_vocabulary_items(
    fitted_pipeline: Pipeline, synthetic_texts_and_labels: tuple[list[str], list[str]]
) -> None:
    explanation = get_explainer("top_tfidf").explain(fitted_pipeline, PODHALE_QUERY)
    assert explanation.method == "top_tfidf"
    assert explanation.attributions
    vocabulary = set(fitted_pipeline.named_steps["tfidf"].get_feature_names_out())
    for attribution in explanation.attributions:
        assert attribution.token in vocabulary


def test_strongest_positive_token_is_a_class_marker(
    fitted_pipeline: Pipeline, synthetic_texts_and_labels: tuple[list[str], list[str]]
) -> None:
    texts, labels = synthetic_texts_and_labels
    explanation = get_explainer("top_tfidf").explain(fitted_pipeline, PODHALE_QUERY)
    assert explanation.predicted_label == "podhale"
    positive = [a for a in explanation.attributions if a.weight > 0]
    assert positive
    strongest = max(positive, key=lambda a: a.weight)
    class_tokens = _class_vocabulary(texts, labels, fitted_pipeline)
    # The strongest evidence for the podhale prediction must be a word that
    # actually occurs in the podhale training texts, and not in standard ones.
    assert strongest.token in class_tokens["podhale"]
    assert strongest.token not in class_tokens["standard"]


def test_attributions_include_negative_evidence(fitted_pipeline: Pipeline) -> None:
    # A pure Podhale sentence has no counter-evidence: every active feature
    # pushes towards the predicted class. Mix in standard-corpus vocabulary so
    # features with negative coefficients for the winner are actually present.
    mixed_query = "Hej, baca się pyto, kaj się owce pasą, a prognoza pogody zapowiada deszcz."
    explanation = get_explainer("top_tfidf").explain(fitted_pipeline, mixed_query)
    weights = [a.weight for a in explanation.attributions]
    assert any(w > 0 for w in weights)
    assert any(w < 0 for w in weights)


def test_top_k_kwarg_caps_each_direction(fitted_pipeline: Pipeline) -> None:
    explanation = get_explainer("top_tfidf", top_k=2).explain(
        fitted_pipeline, PODHALE_QUERY, top_k=1
    )
    positive = [a for a in explanation.attributions if a.weight > 0]
    negative = [a for a in explanation.attributions if a.weight < 0]
    assert len(positive) <= 1
    assert len(negative) <= 1


def test_class_top_features_returns_k_per_class(fitted_pipeline: Pipeline) -> None:
    top = class_top_features(fitted_pipeline, k=5)
    assert set(top) == {"podhale", "silesia", "kurpie", "standard"}
    vocabulary = set(fitted_pipeline.named_steps["tfidf"].get_feature_names_out())
    for label, features in top.items():
        assert len(features) == 5, label
        assert all(feature.token in vocabulary for feature in features)
        weights = [feature.weight for feature in features]
        assert weights == sorted(weights, reverse=True)


def test_feature_union_prefixes_become_readable(
    synthetic_texts_and_labels: tuple[list[str], list[str]],
) -> None:
    texts, labels = synthetic_texts_and_labels
    union = FeatureUnion(
        [
            ("word", TfidfVectorizer()),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))),
        ]
    )
    pipeline = Pipeline(
        [("features", union), ("clf", LogisticRegression(max_iter=2000, random_state=0))]
    ).fit(texts, labels)
    explanation = get_explainer("top_tfidf").explain(pipeline, PODHALE_QUERY)
    assert explanation.attributions
    for attribution in explanation.attributions:
        source, _, feature = attribution.token.partition(": ")
        assert source in {"word", "char"}
        assert feature


def test_calibrated_svm_uses_averaged_coefficients(
    synthetic_texts_and_labels: tuple[list[str], list[str]],
) -> None:
    texts, labels = synthetic_texts_and_labels
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", CalibratedClassifierCV(LinearSVC(random_state=0), cv=2)),
        ]
    ).fit(texts, labels)
    explanation = get_explainer("top_tfidf").explain(pipeline, PODHALE_QUERY)
    assert explanation.attributions
    assert explanation.details["calibrated_average"] is True


def test_non_linear_model_raises_and_points_to_alternatives(
    synthetic_texts_and_labels: tuple[list[str], list[str]],
) -> None:
    texts, labels = synthetic_texts_and_labels
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]
    ).fit(texts, labels)
    with pytest.raises(ConfigurationError, match="lime"):
        get_explainer("top_tfidf").explain(pipeline, PODHALE_QUERY)
    with pytest.raises(ConfigurationError, match="shap"):
        class_top_features(pipeline, k=3)


def test_non_pipeline_input_raises(fitted_pipeline: Pipeline) -> None:
    estimator = fitted_pipeline.named_steps["clf"]
    with pytest.raises(ConfigurationError, match="Pipeline"):
        get_explainer("top_tfidf").explain(estimator, PODHALE_QUERY)


def test_empty_input_raises(fitted_pipeline: Pipeline) -> None:
    with pytest.raises(ConfigurationError, match="empty"):
        get_explainer("top_tfidf").explain(fitted_pipeline, "   ")
