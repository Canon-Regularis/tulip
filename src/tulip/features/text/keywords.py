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
from typing import TYPE_CHECKING, Any

import numpy as np

from tulip.core.exceptions import ConfigurationError
from tulip.features.registries import TEXT_FEATURES
from tulip.features.text._base import DenseTextExtractor, check_per_tokens
from tulip.features.text._resource import read_yaml_resource
from tulip.features.text._tokenize import word_tokens
from tulip.labels.taxonomy import family_for
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "LEXICON_DIALECT_OVERRIDES",
    "DialectKeywordExtractor",
    "canonical_dialect",
    "family_for_lexicon_key",
    "load_lexicon",
]

logger = get_logger(__name__)

#: Name of the bundled starter lexicon (package data under ``lexicons/``).
_BUNDLED_LEXICON = "dialect_markers.yaml"

#: Lexicon keys whose taxonomy ``RegionalDialect`` value differs from the key
#: itself. The lexicon groups general Masovian markers under ``masovia``; the
#: taxonomy's regional-dialect value is ``mazovia_proper`` (its family
#: auto-derives to ``masovian``). This reconciliation is shared by the synthetic
#: generator (which assigns it as a label) and by ``dialect_intensity`` (which
#: groups markers by family), so the mapping lives in exactly one place.
LEXICON_DIALECT_OVERRIDES: dict[str, str] = {"masovia": "mazovia_proper"}


def canonical_dialect(key: str) -> str:
    """Map a lexicon dialect key to its taxonomy ``RegionalDialect`` value."""
    return LEXICON_DIALECT_OVERRIDES.get(key, key)


def family_for_lexicon_key(key: str) -> str | None:
    """Return the dialect-family value for a lexicon key, or ``None`` if unknown.

    Resolves the lexicon key to its canonical regional-dialect value (see
    :data:`LEXICON_DIALECT_OVERRIDES`) and then to its family via
    :func:`tulip.labels.taxonomy.family_for`.
    """
    family = family_for(canonical_dialect(key))
    return family.value if family is not None else None


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
    source, data = read_yaml_resource(path, bundled_name=_BUNDLED_LEXICON, noun="lexicon")
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
        if key in lexicon:
            raise ConfigurationError(
                f"{source}: duplicate dialect key {key!r} after case-folding; "
                "merge the entries under one key"
            )
        lexicon[key] = tuple(dict.fromkeys(cleaned))
    return lexicon


@TEXT_FEATURES.register("dialect_keywords")
class DialectKeywordExtractor(DenseTextExtractor):
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
        check_per_tokens(self.per_tokens)
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
