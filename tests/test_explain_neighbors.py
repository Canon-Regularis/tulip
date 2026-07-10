"""Tests for the nearest_examples cosine-similarity explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from tulip.core.exceptions import ConfigurationError
from tulip.explain import get_explainer
from tulip.explain.neighbors import NearestExamplesExplainer

if TYPE_CHECKING:
    from tulip.core.types import Sample


@pytest.fixture
def indexed_explainer(
    synthetic_samples: list[Sample],
) -> tuple[NearestExamplesExplainer, list[Sample]]:
    texts = [sample.text or "" for sample in synthetic_samples]
    transformer = TfidfVectorizer().fit(texts)
    explainer = NearestExamplesExplainer(k=3)
    explainer.index(synthetic_samples, transformer)
    return explainer, synthetic_samples


def test_exact_duplicate_is_most_similar(
    indexed_explainer: tuple[NearestExamplesExplainer, list[Sample]],
) -> None:
    explainer, samples = indexed_explainer
    query = samples[0]
    explanation = explainer.explain(None, query.text)
    assert explanation.method == "nearest_examples"
    assert explanation.neighbors
    best = explanation.neighbors[0]
    assert best.sample_id == query.id
    assert best.similarity == pytest.approx(1.0, abs=1e-9)
    assert best.label == query.labels.dialect or best.label == query.labels.family


def test_neighbors_sorted_and_k_respected(
    indexed_explainer: tuple[NearestExamplesExplainer, list[Sample]],
) -> None:
    explainer, samples = indexed_explainer
    explanation = explainer.explain(None, samples[5].text)
    assert len(explanation.neighbors) == 3  # constructor default k
    similarities = [neighbor.similarity for neighbor in explanation.neighbors]
    assert similarities == sorted(similarities, reverse=True)

    widened = explainer.explain(None, samples[5].text, k=7)
    assert len(widened.neighbors) == 7
    assert widened.details["k"] == 7


def test_k_larger_than_index_is_clamped(synthetic_samples: list[Sample]) -> None:
    few = synthetic_samples[:4]
    transformer = TfidfVectorizer().fit([sample.text or "" for sample in few])
    explainer = NearestExamplesExplainer(k=100).index(few, transformer)
    explanation = explainer.explain(None, few[0].text)
    assert len(explanation.neighbors) == 4


def test_neighbor_labels_match_gold_dialects(
    indexed_explainer: tuple[NearestExamplesExplainer, list[Sample]],
) -> None:
    explainer, samples = indexed_explainer
    podhale_query = next(s for s in samples if s.labels.dialect == "podhale")
    explanation = explainer.explain(None, podhale_query.text)
    assert explanation.neighbors[0].label == "podhale"


def test_sparse_index_stays_sparse(
    indexed_explainer: tuple[NearestExamplesExplainer, list[Sample]],
) -> None:
    explainer, _ = indexed_explainer
    assert sparse.issparse(explainer._matrix)


def test_dense_transformer_is_supported(synthetic_samples: list[Sample]) -> None:
    class DenseHash:
        """Tiny deterministic dense embedding, no fitting required."""

        def transform(self, texts: list[str]) -> np.ndarray:
            rows = []
            for text in texts:
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                rows.append(rng.normal(size=8))
            return np.vstack(rows)

    explainer = NearestExamplesExplainer(k=2).index(synthetic_samples, DenseHash())
    explanation = explainer.explain(None, synthetic_samples[3].text)
    assert explanation.neighbors[0].sample_id == synthetic_samples[3].id
    assert explanation.neighbors[0].similarity == pytest.approx(1.0, abs=1e-9)


def test_snippets_are_truncated(synthetic_samples: list[Sample]) -> None:
    texts = [sample.text or "" for sample in synthetic_samples]
    transformer = TfidfVectorizer().fit(texts)
    explainer = NearestExamplesExplainer(k=1, snippet_chars=10)
    explainer.index(synthetic_samples, transformer)
    explanation = explainer.explain(None, synthetic_samples[0].text)
    text = explanation.neighbors[0].text or ""
    assert len(text) <= 10 + len("...")


def test_explain_before_index_raises() -> None:
    with pytest.raises(ConfigurationError, match="index"):
        get_explainer("nearest_examples").explain(None, "kaj som owce")


def test_pipeline_fills_predicted_label(
    indexed_explainer: tuple[NearestExamplesExplainer, list[Sample]],
    synthetic_texts_and_labels: tuple[list[str], list[str]],
) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts, labels = synthetic_texts_and_labels
    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=2000))]
    ).fit(texts, labels)
    explainer, samples = indexed_explainer
    explanation = explainer.explain(pipeline, samples[0].text)
    assert explanation.predicted_label in set(labels)
