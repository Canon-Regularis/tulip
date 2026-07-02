"""Dense stylometric features for dialect text.

Registers ``stylometry`` in :data:`tulip.features.registries.TEXT_FEATURES`.
Stylometric statistics complement lexical n-grams: transcribed dialect speech
differs from written standard Polish in sentence rhythm, punctuation habits of
transcribers, and lexical richness, and these signals survive even when the
vocabulary overlaps.

All features are computed per document and are safe on degenerate inputs
(empty strings, single words): every ratio guards its denominator and returns
0.0 instead of dividing by zero.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from tulip.features.registries import TEXT_FEATURES
from tulip.features.text._tokenize import word_tokens

__all__ = ["StylometryExtractor"]

#: Sentence terminators: ., !, ?, and the one-character ellipsis (U+2026).
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+")

#: Three-dot sequences or the one-character ellipsis (U+2026).
_ELLIPSIS_RE = re.compile(r"\.{3}|…")

#: Hyphen-minus, en dash (U+2013), em dash (U+2014).
_DASH_CHARS = "-–—"  # noqa: RUF001  (typographic dashes are the feature being counted)

#: ASCII quotes/apostrophe plus Polish/European typographic quotes:
#: low-9 double (U+201E), right/left double (U+201D/U+201C), guillemets
#: (U+00AB/U+00BB), single curly quotes (U+2018/U+2019), low-9 single (U+201A).
_QUOTE_CHARS = "\"'„”“«»‘’‚"  # noqa: RUF001  (typographic quotes are the feature being counted)

#: Column names, in output order.
_FEATURE_NAMES: tuple[str, ...] = (
    "sentence_count",
    "sentence_length_words_mean",
    "sentence_length_words_std",
    "word_length_mean",
    "word_length_std",
    "commas_per_100_chars",
    "periods_per_100_chars",
    "question_marks_per_100_chars",
    "exclamations_per_100_chars",
    "dashes_per_100_chars",
    "ellipses_per_100_chars",
    "quotes_per_100_chars",
    "digit_ratio",
    "uppercase_ratio",
    "type_token_ratio",
    "hapax_legomena_ratio",
    "yules_k",
)


def _yules_k(token_counts: Counter[str]) -> float:
    """Compute Yule's characteristic K, a length-robust lexical-richness index.

    ``K = 10^4 * (sum_i i^2 * V(i) - N) / N^2`` where ``V(i)`` is the number of
    types occurring ``i`` times and ``N`` the token count. Returns 0.0 for
    empty input (and for all-hapax texts, where the formula is exactly zero).
    """
    n_tokens = sum(token_counts.values())
    if n_tokens == 0:
        return 0.0
    freq_spectrum = Counter(token_counts.values())
    m2 = sum(i * i * v for i, v in freq_spectrum.items())
    return 1e4 * (m2 - n_tokens) / (n_tokens * n_tokens)


def _stylometry_vector(text: str) -> np.ndarray:
    """Compute the stylometric feature vector for one document."""
    n_chars = len(text)
    tokens = word_tokens(text, lowercase=True)
    n_tokens = len(tokens)

    sentence_lengths = [
        length
        for segment in _SENTENCE_SPLIT_RE.split(text)
        if (length := len(word_tokens(segment))) > 0
    ]
    n_sentences = len(sentence_lengths)
    sent_mean = float(np.mean(sentence_lengths)) if n_sentences else 0.0
    sent_std = float(np.std(sentence_lengths)) if n_sentences else 0.0

    word_lengths = [len(token) for token in tokens]
    word_mean = float(np.mean(word_lengths)) if n_tokens else 0.0
    word_std = float(np.std(word_lengths)) if n_tokens else 0.0

    def per_100_chars(count: int) -> float:
        return 100.0 * count / n_chars if n_chars else 0.0

    letters = [ch for ch in text if ch.isalpha()]
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters) if letters else 0.0
    digit_ratio = sum(1 for ch in text if ch.isdigit()) / n_chars if n_chars else 0.0

    counts = Counter(tokens)
    type_token_ratio = len(counts) / n_tokens if n_tokens else 0.0
    hapax_ratio = sum(1 for c in counts.values() if c == 1) / n_tokens if n_tokens else 0.0

    return np.array(
        [
            float(n_sentences),
            sent_mean,
            sent_std,
            word_mean,
            word_std,
            per_100_chars(text.count(",")),
            per_100_chars(text.count(".")),
            per_100_chars(text.count("?")),
            per_100_chars(text.count("!")),
            per_100_chars(sum(text.count(ch) for ch in _DASH_CHARS)),
            per_100_chars(len(_ELLIPSIS_RE.findall(text))),
            per_100_chars(sum(text.count(ch) for ch in _QUOTE_CHARS)),
            digit_ratio,
            uppercase_ratio,
            type_token_ratio,
            hapax_ratio,
            _yules_k(counts),
        ],
        dtype=np.float64,
    )


@TEXT_FEATURES.register("stylometry")
class StylometryExtractor(BaseEstimator, TransformerMixin):
    """Stateless dense stylometric features with named columns.

    Emits one row of 17 features per input document: sentence count,
    mean/std sentence length in words, mean/std word length, punctuation
    frequencies per 100 characters (comma, period, question mark, exclamation
    mark, dash, ellipsis, quotes), digit ratio, uppercase ratio (over letters),
    type-token ratio, hapax legomena ratio, and Yule's K.

    Notes:
        Periods inside a three-dot ellipsis count towards both the period and
        the ellipsis columns; the ellipsis column exists to separate trailing-
        off speech (frequent in transcriptions) from ordinary full stops.
    """

    def fit(self, X: Sequence[str], y: Any = None) -> StylometryExtractor:
        """No-op fit (the extractor is stateless); returns ``self``."""
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Compute stylometric vectors for each document.

        Args:
            X: Sequence of raw text documents.

        Returns:
            Dense float64 array of shape ``(len(X), 17)``.
        """
        documents = list(X)
        if not documents:
            return np.empty((0, len(_FEATURE_NAMES)), dtype=np.float64)
        return np.vstack([_stylometry_vector(str(doc)) for doc in documents])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the 17 stylometric column names, in output order."""
        return np.asarray(_FEATURE_NAMES, dtype=object)
