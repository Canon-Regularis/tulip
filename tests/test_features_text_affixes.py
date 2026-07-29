"""Tests for tulip.features.text.affixes.

The ``affix_frequency`` extractor was registered but only exercised through the
composite; this covers its fit/transform directly so the learned-vocabulary and
per-token-rate logic is guarded on its own.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError


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
from tulip.features.text.affixes import AffixFrequencyExtractor  # noqa: E402


def test_registered_and_shapes() -> None:
    extractor = TEXT_FEATURES.create("affix_frequency")
    assert isinstance(extractor, AffixFrequencyExtractor)
    corpus = ["baca gazda owca", "kowal nowak", "baca kowal"]
    matrix = extractor.fit(corpus).transform(corpus)
    assert matrix.shape == (3, len(extractor.feature_names_))
    assert np.all(np.isfinite(matrix))


def test_feature_names_are_suffix_and_prefix_labelled() -> None:
    extractor = AffixFrequencyExtractor(min_len=2, max_len=2).fit(["baca gazda", "owca baca"])
    assert all(name.startswith(("suffix:-", "prefix:")) for name in extractor.feature_names_)


def test_rate_is_occurrences_over_token_count() -> None:
    # Suffixes only, length 2: "kot" -> "-ot", "pies" -> "-es". Two tokens, one
    # occurrence each, so each rate is 1/2.
    extractor = AffixFrequencyExtractor(min_len=2, max_len=2, include_prefixes=False, min_df=1).fit(
        ["kot pies"]
    )
    names = list(extractor.feature_names_)
    assert set(names) == {"suffix:-ot", "suffix:-es"}
    row = extractor.transform(["kot pies"])[0]
    rates = dict(zip(names, row, strict=True))
    assert rates["suffix:-ot"] == pytest.approx(0.5)
    assert rates["suffix:-es"] == pytest.approx(0.5)


def test_a_whole_word_never_counts_as_its_own_affix() -> None:
    # Every token is exactly min_len long, so no affix is at least one char
    # shorter than its token: the learned vocabulary is empty.
    extractor = AffixFrequencyExtractor(min_len=2, max_len=2).fit(["ab cd", "ef gh"])
    assert extractor.feature_names_ == ()
    assert extractor.transform(["ab cd"]).shape == (1, 0)


def test_empty_document_yields_a_zero_row() -> None:
    extractor = AffixFrequencyExtractor(min_len=2, max_len=3).fit(["baca gazda", "owca kowal"])
    matrix = extractor.transform(["", "   "])
    assert matrix.shape[0] == 2
    assert not matrix.any()


def test_min_df_drops_rare_affixes() -> None:
    # "-ca" appears in both documents (df=2); the doc-unique affixes appear once.
    corpus = ["baca owca", "praca taca"]
    common = AffixFrequencyExtractor(min_len=2, max_len=2, min_df=2, include_prefixes=False).fit(
        corpus
    )
    assert common.feature_names_ == ("suffix:-ca",)


def test_fit_is_deterministic() -> None:
    corpus = ["baca gazda owca", "kowal nowak baca", "praca taca kowal"]
    first = AffixFrequencyExtractor().fit(corpus)
    second = AffixFrequencyExtractor().fit(corpus)
    assert first.feature_names_ == second.feature_names_
    np.testing.assert_array_equal(first.transform(corpus), second.transform(corpus))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_len": 0},
        {"min_len": 3, "max_len": 2},
        {"max_features": 0},
        {"min_df": 0},
        {"include_suffixes": False, "include_prefixes": False},
    ],
)
def test_invalid_params_raise_configuration_error(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        AffixFrequencyExtractor(**kwargs).fit(["baca gazda"])
