"""Tests for the shared feature-union builder (uniform across modalities)."""

from __future__ import annotations

import pytest

from tulip.core.exceptions import ConfigurationError, UnknownComponentError
from tulip.features.audio.composite import build_audio_features
from tulip.features.audio.spectral import MfccExtractor
from tulip.features.text.composite import build_text_features


class TestUniformPolicies:
    def test_text_accepts_bare_names_like_audio(self) -> None:
        union = build_text_features(["char_tfidf", "stylometry"])
        assert [name for name, _ in union.transformer_list] == ["char_tfidf", "stylometry"]

    def test_step_names_normalised_identically_for_both_modalities(self) -> None:
        text_union = build_text_features([{"name": "Char-TFIDF"}])
        audio_union = build_audio_features([{"name": "MFCC"}])
        assert text_union.transformer_list[0][0] == "char_tfidf"
        assert audio_union.transformer_list[0][0] == "mfcc"

    def test_repeated_components_get_numeric_suffixes(self) -> None:
        union = build_text_features(
            [
                {"name": "char_tfidf", "params": {"ngram_range": [2, 3]}},
                {"name": "char_tfidf", "params": {"ngram_range": [4, 5]}},
            ]
        )
        assert [name for name, _ in union.transformer_list] == ["char_tfidf", "char_tfidf_2"]

    def test_empty_configs_raise(self) -> None:
        with pytest.raises(ConfigurationError):
            build_text_features([])
        with pytest.raises(ConfigurationError):
            build_audio_features([])

    def test_unknown_names_raise_with_suggestions(self) -> None:
        with pytest.raises(UnknownComponentError, match="char_tfidf"):
            build_text_features(["char_tfdif"])

    def test_malformed_entry_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="cannot interpret"):
            build_text_features([42])  # type: ignore[list-item]


class TestFitTimeValidation:
    def test_bogus_pooling_stats_rejected_at_fit_not_transform(self) -> None:
        extractor = MfccExtractor(stats=["bogus"])  # sklearn contract: init stores only
        with pytest.raises(ConfigurationError, match="bogus"):
            extractor.fit([])
