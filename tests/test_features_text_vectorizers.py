"""Tests for tulip.features.text.vectorizers (char_tfidf / word_tfidf)."""

from __future__ import annotations

import sys
import types

from sklearn.feature_extraction.text import TfidfVectorizer


def _import_guard() -> None:
    """Keep tulip.features importable before the sibling audio package exists."""
    try:
        import tulip.features
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on build order
        if exc.name != "tulip.features.audio":
            raise
        sys.modules["tulip.features.audio"] = types.ModuleType("tulip.features.audio")
        import tulip.features  # noqa: F401


_import_guard()

from tulip.features.registries import TEXT_FEATURES  # noqa: E402
from tulip.features.text import make_char_tfidf, make_word_tfidf  # noqa: E402


def test_char_tfidf_defaults() -> None:
    vectorizer = make_char_tfidf()
    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.analyzer == "char_wb"
    assert vectorizer.ngram_range == (2, 5)
    assert vectorizer.sublinear_tf is True
    assert vectorizer.min_df == 2
    assert vectorizer.strip_accents is None


def test_char_tfidf_registry_create_with_overrides() -> None:
    vectorizer = TEXT_FEATURES.create("char_tfidf", ngram_range=[3, 4], min_df=1)
    assert vectorizer.ngram_range == (3, 4)  # YAML-style list coerced to tuple
    assert vectorizer.min_df == 1
    assert vectorizer.analyzer == "char_wb"  # untouched defaults survive


def test_word_tfidf_defaults() -> None:
    vectorizer = make_word_tfidf()
    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.analyzer == "word"
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.lowercase is True
    assert vectorizer.strip_accents is None


def test_char_tfidf_fit_transform_shape(
    synthetic_texts_and_labels: tuple[list[str], list[str]],
) -> None:
    texts, _ = synthetic_texts_and_labels
    vectorizer = make_char_tfidf()
    matrix = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    assert matrix.shape == (len(texts), len(names))
    assert matrix.shape[1] > 0


def test_word_tfidf_lowercases_but_preserves_diacritics() -> None:
    texts = ["Godać po naszymu", "godać PO naszymu cołki czas"]
    vectorizer = make_word_tfidf(min_df=1)
    vectorizer.fit(texts)
    vocabulary = set(vectorizer.get_feature_names_out())
    assert "godać" in vocabulary  # lowercased, diacritic intact
    assert "godac" not in vocabulary  # accents never stripped
    assert "po naszymu" in vocabulary  # bigrams present
    assert not any(term != term.lower() for term in vocabulary)


def test_word_tfidf_keeps_single_character_words() -> None:
    vectorizer = make_word_tfidf(min_df=1, ngram_range=(1, 1))
    vectorizer.fit(["u nos w boru", "u nos w lesie"])
    vocabulary = set(vectorizer.get_feature_names_out())
    assert {"u", "w"} <= vocabulary
