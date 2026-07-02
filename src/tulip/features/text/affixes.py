"""Affix (prefix/suffix) frequency features for dialect text.

Registers ``affix_frequency`` in
:data:`tulip.features.registries.TEXT_FEATURES`.

Word-final suffixes are among the strongest written-dialect signals in Polish:
dialects differ systematically in inflectional and derivational endings, e.g.
infinitives in ``-owac`` vs dialectal ``-uwac``, second-person plural/dual
verb endings in ``-ta`` (``robita``, ``widzita``) typical of Greater Poland
and Mazovia, Silesian past-tense forms in ``-och``/``-ech``, and Goralic
``-ymu``/``-ygo`` adjective endings. Word-initial prefixes carry a weaker but
complementary signal (e.g. dialectal ``ober-``, ``wy-``/``wi-`` alternations).
This extractor learns the most document-frequent affixes from training data
and emits their per-document relative frequencies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from tulip.core.exceptions import ConfigurationError
from tulip.features.registries import TEXT_FEATURES
from tulip.features.text._tokenize import word_tokens
from tulip.utils.logging import get_logger

__all__ = ["AffixFrequencyExtractor"]

logger = get_logger(__name__)


@TEXT_FEATURES.register("affix_frequency")
class AffixFrequencyExtractor(BaseEstimator, TransformerMixin):
    """Relative frequencies of learned word-final suffixes and word-initial prefixes.

    The affix vocabulary is learned in :meth:`fit` by document frequency: every
    suffix/prefix of length ``min_len``..``max_len`` (from tokens at least one
    character longer than the affix, so a whole word never counts as its own
    affix) is a candidate, candidates below ``min_df`` documents are dropped,
    and the ``max_features`` most document-frequent survivors (ties broken
    alphabetically) become columns. :meth:`transform` emits, per document, each
    affix's occurrence count divided by the document's token count.

    Feature names are ``"suffix:-<affix>"`` and ``"prefix:<affix>-"``.

    Args:
        min_len: Minimum affix length in characters (>= 1).
        max_len: Maximum affix length in characters (>= ``min_len``).
        max_features: Vocabulary size cap; ``None`` keeps every candidate.
        min_df: Minimum number of fit documents an affix must appear in.
        include_suffixes: Learn word-final suffixes.
        include_prefixes: Learn word-initial prefixes.
        lowercase: Lowercase tokens before extracting affixes.
    """

    def __init__(
        self,
        min_len: int = 2,
        max_len: int = 4,
        max_features: int | None = 300,
        min_df: int = 1,
        include_suffixes: bool = True,
        include_prefixes: bool = True,
        lowercase: bool = True,
    ) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.max_features = max_features
        self.min_df = min_df
        self.include_suffixes = include_suffixes
        self.include_prefixes = include_prefixes
        self.lowercase = lowercase

    def _validate_params(self) -> None:
        """Raise :class:`ConfigurationError` for inconsistent parameters."""
        if self.min_len < 1:
            raise ConfigurationError(f"min_len must be >= 1, got {self.min_len}")
        if self.max_len < self.min_len:
            raise ConfigurationError(
                f"max_len ({self.max_len}) must be >= min_len ({self.min_len})"
            )
        if self.max_features is not None and self.max_features < 1:
            raise ConfigurationError(f"max_features must be >= 1 or None, got {self.max_features}")
        if self.min_df < 1:
            raise ConfigurationError(f"min_df must be >= 1, got {self.min_df}")
        if not (self.include_suffixes or self.include_prefixes):
            raise ConfigurationError("at least one of include_suffixes/include_prefixes required")

    def _doc_affixes(self, text: str) -> Iterator[str]:
        """Yield one affix label per (token, length, side) occurrence in ``text``."""
        for token in word_tokens(text, lowercase=self.lowercase):
            for length in range(self.min_len, self.max_len + 1):
                if len(token) < length + 1:
                    break  # longer affixes cannot fit either
                if self.include_suffixes:
                    yield f"suffix:-{token[-length:]}"
                if self.include_prefixes:
                    yield f"prefix:{token[:length]}-"

    def fit(self, X: Sequence[str], y: Any = None) -> AffixFrequencyExtractor:
        """Learn the affix vocabulary from training documents by document frequency.

        Args:
            X: Sequence of raw text documents.
            y: Ignored (sklearn API compatibility).

        Returns:
            ``self``.

        Raises:
            ConfigurationError: If length/df/feature-count parameters are inconsistent.
        """
        self._validate_params()
        document_frequency: Counter[str] = Counter()
        n_docs = 0
        for text in X:
            n_docs += 1
            document_frequency.update(set(self._doc_affixes(str(text))))
        selected = [a for a, df in document_frequency.items() if df >= self.min_df]
        # Rank by descending document frequency; alphabetical tie-break keeps
        # the selection deterministic across runs and platforms.
        selected.sort(key=lambda affix: (-document_frequency[affix], affix))
        if self.max_features is not None:
            selected = selected[: self.max_features]
        selected.sort()  # stable, readable column order
        self.vocabulary_: dict[str, int] = {affix: i for i, affix in enumerate(selected)}
        self.feature_names_: tuple[str, ...] = tuple(selected)
        self.document_frequency_: dict[str, int] = {a: document_frequency[a] for a in selected}
        if not selected:
            logger.warning(
                "affix_frequency learned an empty vocabulary from %d documents "
                "(min_df=%d); transform will emit zero columns",
                n_docs,
                self.min_df,
            )
        else:
            logger.debug(
                "affix_frequency learned %d affixes from %d documents", len(selected), n_docs
            )
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Compute relative affix frequencies for each document.

        Each cell is the affix's occurrence count in the document divided by
        the document's token count (0.0 for empty documents).

        Args:
            X: Sequence of raw text documents.

        Returns:
            Dense float64 array of shape ``(len(X), len(vocabulary_))``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_fitted()
        documents = list(X)
        matrix = np.zeros((len(documents), len(self.feature_names_)), dtype=np.float64)
        for row, text in enumerate(documents):
            n_tokens = len(word_tokens(str(text), lowercase=self.lowercase))
            if n_tokens == 0:
                continue
            counts = Counter(self._doc_affixes(str(text)))
            for affix, count in counts.items():
                column = self.vocabulary_.get(affix)
                if column is not None:
                    matrix[row, column] = count / n_tokens
        return matrix

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return learned affix column names (``suffix:-...`` / ``prefix:...-``)."""
        self._check_fitted()
        return np.asarray(self.feature_names_, dtype=object)

    def _check_fitted(self) -> None:
        if not hasattr(self, "vocabulary_"):
            raise NotFittedError(
                "This AffixFrequencyExtractor instance is not fitted yet; call fit first."
            )
