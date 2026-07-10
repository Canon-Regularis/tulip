"""Tests for the shared feature-union builder (uniform across modalities)."""

from __future__ import annotations

import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tulip.core.exceptions import ConfigurationError, UnknownComponentError
from tulip.features.audio.composite import build_audio_features
from tulip.features.audio.spectral import MfccExtractor
from tulip.features.text.affixes import AffixFrequencyExtractor
from tulip.features.text.composite import build_text_features
from tulip.features.text.keywords import DialectKeywordExtractor
from tulip.features.text.phonology import PhonologicalMarkerExtractor

#: (extractor class, its own hyper-parameter names) for the estimator-contract tests.
_DENSE_EXTRACTOR_CASES = [
    (DialectKeywordExtractor, {"lexicon_path", "per_tokens"}),
    (PhonologicalMarkerExtractor, {"isogloss_path", "per_tokens"}),
    (
        AffixFrequencyExtractor,
        {
            "min_len",
            "max_len",
            "max_features",
            "min_df",
            "include_suffixes",
            "include_prefixes",
            "lowercase",
        },
    ),
]


class TestDenseExtractorEstimatorContract:
    """The shared _DenseTextExtractor base must not break the sklearn estimator API.

    A base ``__init__`` would shadow the concrete signature that ``get_params`` /
    ``clone`` introspect, silently breaking ``clone()`` and ``GridSearchCV``. The
    base deliberately defines none; these tests pin that.
    """

    @pytest.mark.parametrize(("cls", "params"), _DENSE_EXTRACTOR_CASES)
    def test_get_params_reports_the_concrete_signature(self, cls, params: set) -> None:
        assert set(cls().get_params()) == params

    @pytest.mark.parametrize(("cls", "params"), _DENSE_EXTRACTOR_CASES)
    def test_clone_preserves_params(self, cls, params: set) -> None:
        cloned = clone(cls())
        assert set(cloned.get_params()) == params

    @pytest.mark.parametrize(("cls", "params"), _DENSE_EXTRACTOR_CASES)
    def test_feature_names_before_fit_raises(self, cls, params: set) -> None:
        with pytest.raises(NotFittedError):
            cls().get_feature_names_out()

    def test_distinct_per_tokens_defaults_are_preserved(self) -> None:
        # They differ on purpose: phon columns are dense and unioned with sparse
        # TF-IDF (scale parity, default 1.0); keywords report per-1000 rates.
        assert PhonologicalMarkerExtractor().per_tokens == 1.0
        assert DialectKeywordExtractor().per_tokens == 1000.0


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
