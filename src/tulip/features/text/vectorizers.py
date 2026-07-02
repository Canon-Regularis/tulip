"""TF-IDF vectorizers tuned for Polish dialect text.

Registers two thin, well-defaulted wrappers over
:class:`sklearn.feature_extraction.text.TfidfVectorizer` in
:data:`tulip.features.registries.TEXT_FEATURES`:

- ``char_tfidf``: character n-grams within word boundaries (``char_wb``,
  2-5 grams). Character n-grams capture sub-word orthographic reflexes of
  dialect phonology (mazurzenie respellings, o/uo alternations, dialectal
  suffix shapes) and are robust to rich Polish inflection.
- ``word_tfidf``: word 1-2 grams, lowercased but with diacritics preserved --
  diacritics are part of the dialect signal and must never be stripped.

Every default is overridable through registry ``params``.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from tulip.features.registries import TEXT_FEATURES

__all__ = ["make_char_tfidf", "make_word_tfidf"]

#: Defaults for ``char_tfidf``. ``char_wb`` pads n-grams with spaces at word
#: edges, which makes word-final suffixes and word-initial prefixes explicit
#: n-grams -- exactly where Polish dialect morphology lives. ``sublinear_tf``
#: dampens length effects in transcribed speech; ``min_df=2`` drops one-off
#: transcription noise.
_CHAR_TFIDF_DEFAULTS: dict[str, Any] = {
    "analyzer": "char_wb",
    "ngram_range": (2, 5),
    "sublinear_tf": True,
    "min_df": 2,
    "lowercase": True,
    # Diacritics are the signal (e.g. dialectal o/ó, e/y alternations): never strip.
    "strip_accents": None,
}

#: Defaults for ``word_tfidf``. The token pattern keeps single-character words
#: (Polish prepositions and dialect particles), unlike sklearn's default.
_WORD_TFIDF_DEFAULTS: dict[str, Any] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 2,
    "lowercase": True,
    "strip_accents": None,
    "token_pattern": r"(?u)\b\w+\b",
}


def _merged(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge overrides into defaults, coercing YAML-style list n-gram ranges."""
    params = {**defaults, **overrides}
    ngram_range = params.get("ngram_range")
    if isinstance(ngram_range, list):
        # YAML configs express tuples as lists; TfidfVectorizer wants a tuple.
        params["ngram_range"] = tuple(ngram_range)
    return params


@TEXT_FEATURES.register("char_tfidf")
def make_char_tfidf(**overrides: Any) -> TfidfVectorizer:
    """Create the ``char_tfidf`` extractor (char_wb 2-5 gram TF-IDF).

    Args:
        **overrides: Any :class:`TfidfVectorizer` keyword, overriding the
            defaults (``analyzer="char_wb"``, ``ngram_range=(2, 5)``,
            ``sublinear_tf=True``, ``min_df=2``). ``ngram_range`` may be given
            as a two-element list (YAML-friendly).

    Returns:
        An unfitted :class:`TfidfVectorizer`.
    """
    return TfidfVectorizer(**_merged(_CHAR_TFIDF_DEFAULTS, overrides))


@TEXT_FEATURES.register("word_tfidf")
def make_word_tfidf(**overrides: Any) -> TfidfVectorizer:
    """Create the ``word_tfidf`` extractor (word 1-2 gram TF-IDF).

    Text is lowercased but diacritics are preserved (``strip_accents=None``):
    forms like ``godać`` vs ``gadać`` must stay distinct.

    Args:
        **overrides: Any :class:`TfidfVectorizer` keyword, overriding the
            defaults (``ngram_range=(1, 2)``, ``sublinear_tf=True``,
            ``min_df=2``). ``ngram_range`` may be given as a two-element list.

    Returns:
        An unfitted :class:`TfidfVectorizer`.
    """
    return TfidfVectorizer(**_merged(_WORD_TFIDF_DEFAULTS, overrides))
