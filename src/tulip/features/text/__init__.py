"""Text feature extraction for Polish dialect classification.

Importing this package registers the built-in text feature extractors in
:data:`tulip.features.registries.TEXT_FEATURES` under their canonical names:
``char_tfidf``, ``word_tfidf``, ``stylometry``, ``affix_frequency``, and
``dialect_keywords``. All extractors are scikit-learn transformers operating
on sequences of strings; :func:`build_text_features` composes any subset into
a single :class:`sklearn.pipeline.FeatureUnion`.

The bundled dialect-keyword lexicon ships as package data under
``lexicons/dialect_markers.yaml`` and is loaded via ``importlib.resources``.
"""

from __future__ import annotations

from tulip.features.text.affixes import AffixFrequencyExtractor
from tulip.features.text.composite import build_text_features
from tulip.features.text.keywords import DialectKeywordExtractor, load_lexicon
from tulip.features.text.stylometry import StylometryExtractor
from tulip.features.text.vectorizers import make_char_tfidf, make_word_tfidf

__all__ = [
    "AffixFrequencyExtractor",
    "DialectKeywordExtractor",
    "StylometryExtractor",
    "build_text_features",
    "load_lexicon",
    "make_char_tfidf",
    "make_word_tfidf",
]
