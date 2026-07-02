"""Tests for tulip.features.text.stylometry."""

from __future__ import annotations

import sys
import types

import numpy as np


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
from tulip.features.text import StylometryExtractor  # noqa: E402


def _row(text: str) -> dict[str, float]:
    extractor = StylometryExtractor().fit([])
    values = extractor.transform([text])[0]
    names = extractor.get_feature_names_out()
    return dict(zip(names, values, strict=True))


def test_registered_and_shapes_on_corpus(
    synthetic_texts_and_labels: tuple[list[str], list[str]],
) -> None:
    texts, _ = synthetic_texts_and_labels
    extractor = TEXT_FEATURES.create("stylometry")
    matrix = extractor.fit(texts).transform(texts)
    names = extractor.get_feature_names_out()
    assert matrix.shape == (len(texts), len(names))
    assert len(names) == 17
    assert np.all(np.isfinite(matrix))


def test_hand_computed_micro_input() -> None:
    # 24 chars; tokens: ala ma kota ala ma psa; 2 sentences of 3 words each.
    row = _row("Ala ma kota. Ala ma psa!")
    assert row["sentence_count"] == 2.0
    assert row["sentence_length_words_mean"] == 3.0
    assert row["sentence_length_words_std"] == 0.0
    assert np.isclose(row["word_length_mean"], 17 / 6)
    assert np.isclose(row["word_length_std"], np.std([3, 2, 4, 3, 2, 3]))
    assert row["commas_per_100_chars"] == 0.0
    assert np.isclose(row["periods_per_100_chars"], 100 / 24)
    assert row["question_marks_per_100_chars"] == 0.0
    assert np.isclose(row["exclamations_per_100_chars"], 100 / 24)
    assert row["digit_ratio"] == 0.0
    assert np.isclose(row["uppercase_ratio"], 2 / 17)  # A, A among 17 letters
    assert np.isclose(row["type_token_ratio"], 4 / 6)
    assert np.isclose(row["hapax_legomena_ratio"], 2 / 6)  # kota, psa
    # Counts: ala:2 ma:2 kota:1 psa:1 -> M2 = 4*2 + 1*2 = 10, N = 6.
    assert np.isclose(row["yules_k"], 1e4 * (10 - 6) / 36)


def test_ellipsis_dash_quote_and_digit_counts() -> None:
    text = 'No i co... "tak" — 12'  # 21 chars
    row = _row(text)
    assert len(text) == 21
    assert np.isclose(row["ellipses_per_100_chars"], 100 / 21)
    assert np.isclose(row["periods_per_100_chars"], 3 * 100 / 21)  # dots inside ellipsis
    assert np.isclose(row["quotes_per_100_chars"], 2 * 100 / 21)
    assert np.isclose(row["dashes_per_100_chars"], 100 / 21)
    assert np.isclose(row["digit_ratio"], 2 / 21)


def test_yules_k_repeated_token() -> None:
    # N = 4, one type with frequency 4: K = 1e4 * (16 - 4) / 16 = 7500.
    assert np.isclose(_row("ba ba ba ba")["yules_k"], 7500.0)


def test_degenerate_inputs_are_all_finite() -> None:
    extractor = StylometryExtractor()
    matrix = extractor.transform(["", "   ", "?!...", "x"])
    assert matrix.shape == (4, 17)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix[0] == 0.0)  # empty string: every feature guarded to 0


def test_text_without_terminator_is_one_sentence() -> None:
    row = _row("ala ma kota")
    assert row["sentence_count"] == 1.0
    assert row["sentence_length_words_mean"] == 3.0


def test_empty_input_sequence() -> None:
    matrix = StylometryExtractor().transform([])
    assert matrix.shape == (0, 17)
