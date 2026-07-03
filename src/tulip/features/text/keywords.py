"""Lexicon-based dialect keyword features.

Registers ``dialect_keywords`` in
:data:`tulip.features.registries.TEXT_FEATURES`.

The extractor counts occurrences of known dialect-marker lexemes (e.g. Podhale
``baca``, Silesian ``gryfny``, Kashubian ``chëcz``) and normalises the counts
per 1000 tokens, yielding one column per dialect plus a total. Matching is
whole-word, case-insensitive, and diacritic-exact: ``godac`` does not match
``godać``, because the diacritic is precisely what distinguishes the dialectal
form.

A starter lexicon ships as package data
(``tulip/features/text/lexicons/dialect_markers.yaml``); pass
``lexicon_path`` to extend or replace it. The lexicon format is a YAML mapping
from dialect key to a list of single-word markers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from tulip.core.exceptions import ConfigurationError
from tulip.features.registries import TEXT_FEATURES
from tulip.features.text._tokenize import word_tokens
from tulip.utils.logging import get_logger

__all__ = ["DialectKeywordExtractor", "load_lexicon"]

logger = get_logger(__name__)

#: Name of the bundled starter lexicon (package data under ``lexicons/``).
_BUNDLED_LEXICON = "dialect_markers.yaml"


def _bundled_lexicon_text() -> str:
    """Read the bundled starter lexicon via importlib.resources (zip/Windows-safe)."""
    resource = (
        resources.files("tulip.features.text").joinpath("lexicons").joinpath(_BUNDLED_LEXICON)
    )
    return resource.read_text(encoding="utf-8")


def load_lexicon(path: str | Path | None = None) -> dict[str, tuple[str, ...]]:
    """Load and validate a dialect-marker lexicon.

    Args:
        path: Path to a YAML lexicon file (mapping ``dialect -> [markers]``).
            ``None`` loads the bundled starter lexicon.

    Returns:
        Mapping from lowercased dialect key to a deduplicated tuple of
        lowercased single-word markers (diacritics preserved).

    Raises:
        ConfigurationError: If the file is missing or the lexicon is not a
            non-empty mapping of dialect keys to lists of single-word strings.
    """
    if path is None:
        source = f"bundled lexicon {_BUNDLED_LEXICON!r}"
        raw = _bundled_lexicon_text()
    else:
        lexicon_path = Path(path)
        source = str(lexicon_path)
        if not lexicon_path.is_file():
            raise ConfigurationError(f"lexicon file not found: {lexicon_path}")
        raw = lexicon_path.read_text(encoding="utf-8")

    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or not data:
        raise ConfigurationError(
            f"{source}: lexicon must be a non-empty mapping of dialect -> list of markers"
        )
    lexicon: dict[str, tuple[str, ...]] = {}
    for dialect, markers in data.items():
        key = str(dialect).strip().lower()
        if not key:
            raise ConfigurationError(f"{source}: empty dialect key")
        if not isinstance(markers, list):
            raise ConfigurationError(f"{source}: markers for {key!r} must be a list")
        cleaned: list[str] = []
        for marker in markers:
            if not isinstance(marker, str):
                raise ConfigurationError(
                    f"{source}: marker {marker!r} for {key!r} is not a string "
                    "(quote YAML-ambiguous words such as 'no' or 'tak')"
                )
            normalised = marker.strip().lower()
            if not normalised:
                continue
            if any(ch.isspace() for ch in normalised):
                raise ConfigurationError(
                    f"{source}: marker {marker!r} for {key!r} contains whitespace; "
                    "only single-word markers are supported"
                )
            cleaned.append(normalised)
        if not cleaned:
            raise ConfigurationError(f"{source}: dialect {key!r} has no usable markers")
        lexicon[key] = tuple(dict.fromkeys(cleaned))
    return lexicon


@TEXT_FEATURES.register("dialect_keywords")
class DialectKeywordExtractor(TransformerMixin, BaseEstimator):
    """Per-dialect counts of known dialect-marker lexemes, per 1000 tokens.

    Emits one column per lexicon dialect (``keywords:<dialect>``, dialects in
    sorted order) plus ``keywords:total``. Each dialect column is the number
    of tokens matching that dialect's markers, scaled to ``per_tokens`` tokens
    (default: per 1000). The total is the sum of the dialect columns; markers
    attested in several dialects (e.g. ``kaj`` in both Podhale and Silesia)
    count towards each.

    Matching is whole-word (letter-run tokens), case-insensitive, and
    diacritic-exact.

    Args:
        lexicon_path: YAML lexicon file replacing the bundled starter lexicon
            (see :func:`load_lexicon` for the format); ``None`` uses the
            bundled one.
        per_tokens: Normalisation base; counts are scaled to this many tokens.
    """

    def __init__(
        self,
        lexicon_path: str | Path | None = None,
        per_tokens: float = 1000.0,
    ) -> None:
        self.lexicon_path = lexicon_path
        self.per_tokens = per_tokens

    def fit(self, X: Sequence[str], y: Any = None) -> DialectKeywordExtractor:
        """Load the lexicon and freeze the column layout.

        Args:
            X: Ignored (the lexicon, not the data, defines the features).
            y: Ignored (sklearn API compatibility).

        Returns:
            ``self``.

        Raises:
            ConfigurationError: If ``per_tokens`` is not positive or the
                lexicon is missing/malformed.
        """
        if self.per_tokens <= 0:
            raise ConfigurationError(f"per_tokens must be > 0, got {self.per_tokens}")
        self.lexicon_: dict[str, tuple[str, ...]] = load_lexicon(self.lexicon_path)
        self.dialects_: tuple[str, ...] = tuple(sorted(self.lexicon_))
        self.marker_sets_: dict[str, frozenset[str]] = {
            dialect: frozenset(self.lexicon_[dialect]) for dialect in self.dialects_
        }
        self.feature_names_: tuple[str, ...] = (
            *(f"keywords:{dialect}" for dialect in self.dialects_),
            "keywords:total",
        )
        logger.debug(
            "dialect_keywords loaded %d dialects, %d markers total",
            len(self.dialects_),
            sum(len(markers) for markers in self.lexicon_.values()),
        )
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Compute normalised marker counts for each document.

        Args:
            X: Sequence of raw text documents.

        Returns:
            Dense float64 array of shape ``(len(X), n_dialects + 1)``; all
            zeros for documents with no tokens.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_fitted()
        documents = list(X)
        matrix = np.zeros((len(documents), len(self.feature_names_)), dtype=np.float64)
        for row, text in enumerate(documents):
            tokens = word_tokens(str(text), lowercase=True)
            if not tokens:
                continue
            counts = Counter(tokens)
            scale = self.per_tokens / len(tokens)
            total = 0
            for column, dialect in enumerate(self.dialects_):
                hits = sum(counts[m] for m in self.marker_sets_[dialect] if m in counts)
                matrix[row, column] = hits * scale
                total += hits
            matrix[row, -1] = total * scale
        return matrix

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return ``keywords:<dialect>`` column names plus ``keywords:total``."""
        self._check_fitted()
        return np.asarray(self.feature_names_, dtype=object)

    def _check_fitted(self) -> None:
        if not hasattr(self, "feature_names_"):
            raise NotFittedError(
                "This DialectKeywordExtractor instance is not fitted yet; call fit first."
            )
