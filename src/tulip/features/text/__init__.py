"""Text feature extraction for Polish dialect classification.

Importing this package registers the built-in text feature extractors in
:data:`tulip.features.registries.TEXT_FEATURES` under their canonical names:
``char_tfidf``, ``word_tfidf``, ``stylometry``, ``affix_frequency``,
``dialect_keywords``, and ``phonological_markers``. All extractors are
scikit-learn transformers operating on sequences of strings;
:func:`build_text_features` composes any subset into a single
:class:`sklearn.pipeline.FeatureUnion`.

Bundled lexicons ship as package data under ``lexicons/`` (the whole-word
``dialect_markers.yaml`` and the sub-lexical ``isoglosses.yaml``) and are loaded
via ``importlib.resources``.
"""

from __future__ import annotations

from tulip.features.text.affixes import AffixFrequencyExtractor
from tulip.features.text.composite import build_text_features
from tulip.features.text.keywords import DialectKeywordExtractor, load_lexicon
from tulip.features.text.phonology import (
    DigraphRate,
    IsoglossPattern,
    PhonologicalFeature,
    PhonologicalMarkerExtractor,
    load_isoglosses,
)
from tulip.features.text.stylometry import StylometryExtractor
from tulip.features.text.vectorizers import make_char_tfidf, make_word_tfidf

__all__ = [
    "AffixFrequencyExtractor",
    "DialectKeywordExtractor",
    "DigraphRate",
    "IsoglossPattern",
    "PhonologicalFeature",
    "PhonologicalMarkerExtractor",
    "StylometryExtractor",
    "build_text_features",
    "load_isoglosses",
    "load_lexicon",
    "make_char_tfidf",
    "make_word_tfidf",
]
