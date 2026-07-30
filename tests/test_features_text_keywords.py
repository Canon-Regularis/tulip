"""Tests for tulip.features.text.keywords.

Covers the dialect-marker lexicon loader (validation and normalisation) and the
``dialect_keywords`` extractor's per-1000-token counting, including the
whole-word, case-insensitive, diacritic-exact matching the module promises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import ensure_features_importable
from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path


ensure_features_importable()

from tulip.features.registries import TEXT_FEATURES  # noqa: E402
from tulip.features.text.keywords import (  # noqa: E402
    DialectKeywordExtractor,
    canonical_dialect,
    family_for_lexicon_key,
    load_lexicon,
)


def _lexicon(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "lexicon.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------- load_lexicon


def test_load_lexicon_normalises_lowercases_and_dedups(tmp_path: Path) -> None:
    path = _lexicon(tmp_path, "Podhale:\n  - Baca\n  - baca\n  - kaj\nSilesia:\n  - Gynau\n")
    lex = load_lexicon(path)
    assert lex == {"podhale": ("baca", "kaj"), "silesia": ("gynau",)}


def test_load_bundled_lexicon_is_non_empty() -> None:
    lex = load_lexicon(None)
    assert lex and all(isinstance(markers, tuple) and markers for markers in lex.values())


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("[]\n", "non-empty mapping"),
        ("podhale: not-a-list\n", "must be a list"),
        ("podhale:\n  - 42\n", "not a string"),
        ("podhale:\n  - two words\n", "whitespace"),
        ("podhale:\n  - '   '\n", "no usable markers"),
        ("Podhale:\n  - baca\npodhale:\n  - kaj\n", "duplicate"),
    ],
)
def test_load_lexicon_rejects_bad_input(tmp_path: Path, body: str, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        load_lexicon(_lexicon(tmp_path, body))


# ------------------------------------------------------------------ transform


def _fit(tmp_path: Path, body: str, per_tokens: float = 1000.0) -> DialectKeywordExtractor:
    return DialectKeywordExtractor(
        lexicon_path=_lexicon(tmp_path, body), per_tokens=per_tokens
    ).fit([])


def test_registered_and_feature_names(tmp_path: Path) -> None:
    extractor = TEXT_FEATURES.create("dialect_keywords")
    assert isinstance(extractor, DialectKeywordExtractor)
    fitted = _fit(tmp_path, "silesia:\n  - gynau\npodhale:\n  - baca\n")
    # dialects sorted, then a total column
    assert fitted.feature_names_ == ("keywords:podhale", "keywords:silesia", "keywords:total")


def test_counts_are_scaled_per_thousand_tokens(tmp_path: Path) -> None:
    extractor = _fit(tmp_path, "podhale:\n  - baca\nsilesia:\n  - kaj\n")
    # 3 tokens: baca x2 (podhale), kaj x1 (silesia). scale = 1000/3.
    row = extractor.transform(["baca baca kaj"])[0]
    values = dict(zip(extractor.feature_names_, row, strict=True))
    assert values["keywords:podhale"] == pytest.approx(2000 / 3)
    assert values["keywords:silesia"] == pytest.approx(1000 / 3)
    assert values["keywords:total"] == pytest.approx(1000.0)


def test_marker_shared_by_two_dialects_counts_towards_each(tmp_path: Path) -> None:
    extractor = _fit(tmp_path, "podhale:\n  - kaj\nsilesia:\n  - kaj\n")
    row = extractor.transform(["kaj"])[0]  # 1 token, scale 1000
    values = dict(zip(extractor.feature_names_, row, strict=True))
    assert values["keywords:podhale"] == pytest.approx(1000.0)
    assert values["keywords:silesia"] == pytest.approx(1000.0)
    assert values["keywords:total"] == pytest.approx(2000.0)  # summed, not deduped


def test_matching_is_diacritic_exact(tmp_path: Path) -> None:
    extractor = _fit(tmp_path, "podhale:\n  - godać\n")
    row = extractor.transform(["godac godać"])[0]  # only the accented token matches
    values = dict(zip(extractor.feature_names_, row, strict=True))
    assert values["keywords:podhale"] == pytest.approx(500.0)  # 1 of 2 tokens, x1000


def test_matching_is_whole_word_and_case_insensitive(tmp_path: Path) -> None:
    extractor = _fit(tmp_path, "podhale:\n  - baca\n")
    # BACA matches (case-insensitive); bacan/abaca are different tokens (whole-word).
    row = extractor.transform(["BACA bacan abaca Baca"])[0]
    values = dict(zip(extractor.feature_names_, row, strict=True))
    assert values["keywords:podhale"] == pytest.approx(2 * 1000 / 4)  # 2 of 4 tokens


def test_empty_and_tokenless_documents_are_zero_rows(tmp_path: Path) -> None:
    extractor = _fit(tmp_path, "podhale:\n  - baca\n")
    matrix = extractor.transform(["", "   ", "!!! ???"])
    assert matrix.shape == (3, 2)
    assert not matrix.any()


def test_per_tokens_must_be_positive_and_finite(tmp_path: Path) -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ConfigurationError):
            _fit(tmp_path, "podhale:\n  - baca\n", per_tokens=bad)


# ------------------------------------------------------- taxonomy reconciliation


def test_canonical_dialect_applies_the_masovia_override() -> None:
    assert canonical_dialect("masovia") == "mazovia_proper"
    assert canonical_dialect("podhale") == "podhale"  # unchanged when no override


def test_family_for_lexicon_key_resolves_through_the_override() -> None:
    # masovia -> mazovia_proper -> its family (masovian); an unknown key -> None.
    assert family_for_lexicon_key("masovia") == "masovian"
    assert family_for_lexicon_key("not_a_real_dialect_xyz") is None
