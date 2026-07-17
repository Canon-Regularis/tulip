"""Tests for the classical baseline model factories (``tulip.models.classical``)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from tulip.core.exceptions import ConfigurationError, MissingDependencyError


def _stub_missing_model_siblings() -> None:
    """Allow importing ``tulip.models`` while sibling modules are unwritten.

    ``tulip/models/__init__.py`` eagerly imports every built-in model module
    for registration; during parallel development some siblings may not exist
    yet. Injecting empty placeholder modules keeps this test module runnable
    and is a no-op once the real files exist.
    """
    spec = importlib.util.find_spec("tulip")
    assert spec is not None and spec.origin is not None
    models_dir = Path(spec.origin).parent / "models"
    for sibling in ("classical", "fasttext_model", "neural_audio", "neural_text"):
        if not (models_dir / f"{sibling}.py").exists():
            name = f"tulip.models.{sibling}"
            sys.modules.setdefault(name, types.ModuleType(name))


_stub_missing_model_siblings()

from tulip.models import MODELS  # noqa: E402
from tulip.models.classical import DEFAULT_SEED, LabelEncodedClassifier  # noqa: E402

CORE_MODELS = ("naive_bayes", "logistic_regression", "linear_svm", "random_forest")
BOOSTED_MODELS = ("xgboost", "lightgbm")
#: The conftest corpus has four balanced classes.
CHANCE = 0.25


@pytest.fixture
def tfidf_corpus(
    synthetic_texts_and_labels: tuple[list[str], list[str]],
) -> tuple[object, np.ndarray]:
    """Character TF-IDF matrix and label array over the synthetic corpus."""
    texts, labels = synthetic_texts_and_labels
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts)
    return matrix, np.asarray(labels)


def test_all_canonical_names_registered() -> None:
    for name in (*CORE_MODELS, *BOOSTED_MODELS):
        assert name in MODELS, f"{name!r} missing from MODELS registry"


@pytest.mark.parametrize("name", CORE_MODELS)
def test_core_model_trains_better_than_chance(
    name: str, tfidf_corpus: tuple[object, np.ndarray]
) -> None:
    matrix, labels = tfidf_corpus
    model = MODELS.create(name, random_state=0)
    fitted = model.fit(matrix, labels)
    assert fitted is model
    accuracy = float(np.mean(model.predict(matrix) == labels))
    assert accuracy > 2 * CHANCE


@pytest.mark.parametrize("name", CORE_MODELS)
def test_core_model_probabilities_and_classes(
    name: str, tfidf_corpus: tuple[object, np.ndarray]
) -> None:
    matrix, labels = tfidf_corpus
    model = MODELS.create(name, random_state=0).fit(matrix, labels)
    proba = model.predict_proba(matrix)
    assert proba.shape == (matrix.shape[0], len(set(labels)))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0.0)
    assert all(isinstance(label, str) for label in model.classes_)
    assert sorted(model.classes_) == sorted(set(labels))


def test_majority_registered() -> None:
    assert "majority" in MODELS


def test_majority_predicts_most_frequent_and_returns_priors() -> None:
    # Five 'a', two 'b', one 'c': the majority is 'a'. Features are ignored.
    features = np.zeros((8, 3))
    labels = np.array(["a"] * 5 + ["b"] * 2 + ["c"] * 1)
    model = MODELS.create("majority", random_state=0).fit(features, labels)

    predictions = model.predict(np.zeros((4, 3)))
    assert set(predictions) == {"a"}

    proba = model.predict_proba(np.zeros((4, 3)))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.allclose(proba, proba[0])  # constant priors across samples
    classes = list(model.classes_)
    assert classes == ["a", "b", "c"]  # sklearn sorts the labels
    assert proba[0][classes.index("a")] == pytest.approx(5 / 8)


def test_linear_svm_exposes_predict_proba(tfidf_corpus: tuple[object, np.ndarray]) -> None:
    matrix, labels = tfidf_corpus
    model = MODELS.create("linear_svm", seed=0)
    assert callable(getattr(model, "predict_proba", None))
    model.fit(matrix, labels)
    proba = model.predict_proba(matrix)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_overridden_params_take_effect() -> None:
    forest = MODELS.create("random_forest", n_estimators=10, random_state=3)
    assert forest.n_estimators == 10
    assert forest.random_state == 3

    bayes = MODELS.create("naive_bayes", alpha=1.5)
    assert bayes.alpha == pytest.approx(1.5)

    logreg = MODELS.create("logistic_regression", C=0.5, seed=7)
    assert pytest.approx(0.5) == logreg.C
    assert logreg.random_state == 7
    assert logreg.max_iter == 2000  # untouched defaults are retained

    svm = MODELS.create("linear_svm", cv=5, method="isotonic", C=2.0, seed=1)
    assert svm.cv == 5
    assert svm.method == "isotonic"
    assert pytest.approx(2.0) == svm.estimator.C
    assert svm.estimator.random_state == 1
    assert svm.estimator.class_weight == "balanced"


def test_default_seed_applied() -> None:
    assert MODELS.create("random_forest").random_state == DEFAULT_SEED
    assert MODELS.create("logistic_regression").random_state == DEFAULT_SEED


def test_matching_seed_spellings_allowed() -> None:
    assert MODELS.create("random_forest", random_state=5, seed=5).random_state == 5


def test_conflicting_seed_spellings_raise() -> None:
    with pytest.raises(ConfigurationError):
        MODELS.create("random_forest", random_state=1, seed=2)


def test_naive_bayes_accepts_and_ignores_seed() -> None:
    model = MODELS.create("naive_bayes", seed=3)
    assert model.alpha == pytest.approx(0.1)
    assert not hasattr(model, "random_state")


def test_random_forest_deterministic_under_fixed_seed(
    tfidf_corpus: tuple[object, np.ndarray],
) -> None:
    matrix, labels = tfidf_corpus
    first = MODELS.create("random_forest", n_estimators=20, seed=11).fit(matrix, labels)
    second = MODELS.create("random_forest", n_estimators=20, seed=11).fit(matrix, labels)
    np.testing.assert_array_equal(first.predict(matrix), second.predict(matrix))
    np.testing.assert_array_equal(first.predict_proba(matrix), second.predict_proba(matrix))


def test_label_encoded_classifier_roundtrips_labels(
    tfidf_corpus: tuple[object, np.ndarray],
) -> None:
    from sklearn.linear_model import LogisticRegression

    matrix, labels = tfidf_corpus
    model = LabelEncodedClassifier(LogisticRegression(max_iter=500, random_state=0))
    model.fit(matrix, labels)
    assert list(model.classes_) == sorted(set(labels))
    predictions = model.predict(matrix)
    assert set(predictions) <= set(labels)
    proba = model.predict_proba(matrix)
    assert proba.shape == (matrix.shape[0], len(model.classes_))
    # Probability columns follow classes_ order: argmax must agree with predict.
    np.testing.assert_array_equal(model.classes_[np.argmax(proba, axis=1)], predictions)


def test_label_encoded_classifier_nested_params() -> None:
    from sklearn.linear_model import LogisticRegression

    wrapped = LabelEncodedClassifier(LogisticRegression(C=1.0))
    assert wrapped.get_params()["estimator__C"] == pytest.approx(1.0)
    wrapped.set_params(estimator__C=0.25)
    assert pytest.approx(0.25) == wrapped.estimator.C
    with pytest.raises(ConfigurationError):
        wrapped.set_params(bogus=1)


@pytest.mark.parametrize(("name", "module"), [("xgboost", "xgboost"), ("lightgbm", "lightgbm")])
def test_boosted_factory_raises_missing_dependency(
    name: str, module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None entry in sys.modules forces ImportError even when installed.
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(MissingDependencyError, match="boosting") as excinfo:
        MODELS.create(name)
    assert excinfo.value.module == module
    assert excinfo.value.extra == "boosting"


def test_xgboost_trains_with_string_labels(tfidf_corpus: tuple[object, np.ndarray]) -> None:
    pytest.importorskip("xgboost")
    matrix, labels = tfidf_corpus
    model = MODELS.create("xgboost", n_estimators=20, seed=0)
    assert isinstance(model, LabelEncodedClassifier)
    assert model.estimator.n_estimators == 20  # override reached the inner estimator
    model.fit(matrix, labels)
    assert all(isinstance(label, str) for label in model.classes_)
    predictions = model.predict(matrix)
    assert set(predictions) <= set(labels)
    assert float(np.mean(predictions == labels)) > 2 * CHANCE
    proba = model.predict_proba(matrix)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lightgbm_trains_with_string_labels(tfidf_corpus: tuple[object, np.ndarray]) -> None:
    pytest.importorskip("lightgbm")
    matrix, labels = tfidf_corpus
    # min_child_samples=1 lets trees split on this deliberately tiny corpus.
    model = MODELS.create("lightgbm", n_estimators=30, min_child_samples=1, seed=0)
    assert isinstance(model, LabelEncodedClassifier)
    assert model.estimator.n_estimators == 30
    model.fit(matrix, labels)
    assert all(isinstance(label, str) for label in model.classes_)
    predictions = model.predict(matrix)
    assert set(predictions) <= set(labels)
    assert float(np.mean(predictions == labels)) > 2 * CHANCE
    proba = model.predict_proba(matrix)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
