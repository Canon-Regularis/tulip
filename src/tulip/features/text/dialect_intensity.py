"""Continuous dialect-intensity score: one bounded [0, 1] "how dialectal" index.

Registers ``dialect_intensity`` in
:data:`tulip.features.registries.TEXT_FEATURES`.

The lexical and phonological features answer *which* dialect; this one answers
*how dialectal at all*: a single interpretable number, plus a per-family
breakdown, anchored so that **0 means indistinguishable from standard Polish**.
It is useful wherever a scalar dialectality signal is wanted: gating
self-training pseudo-labels on genuine dialect signal rather than model
over-confidence, triaging noisy corpora, driving map opacity, or as a compact
dense feature.

It is pure composition: it re-implements no detection. Lexical evidence comes
from the shared marker lexicon (:func:`tulip.features.text.keywords.load_lexicon`,
grouped into families by
:func:`~tulip.features.text.keywords.family_for_lexicon_key`); phonological
evidence comes from the *fired* rate of the detectable rewrite rules
(:func:`tulip.features.text.phonological_rules.load_phonological_rules`). The two
densities are fused and squashed to [0, 1] by a fixed, documented saturating
map, so the score is deterministic and needs no fitting.

Why the anchor needs no separate "standard reference" corpus: standard Polish is
*definitionally* the absence of dialectal evidence: no marker lexeme, no fired
sound change, so a standard text has zero signal and maps to exactly 0. The
diacritic-exact marker matching and the rules' exclusion stoplists keep standard
text from accruing spurious signal, so the zero anchor holds without an external
baseline.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from tulip.core.exceptions import ConfigurationError
from tulip.features.registries import TEXT_FEATURES
from tulip.features.text._base import DenseTextExtractor
from tulip.features.text._tokenize import word_tokens
from tulip.features.text.keywords import family_for_lexicon_key, load_lexicon
from tulip.features.text.phonological_rules import load_phonological_rules
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tulip.features.text.phonological_rules import PhonologicalRule

__all__ = ["DEFAULT_FIRED_WEIGHT", "DEFAULT_MARKER_WEIGHT", "DialectIntensityExtractor"]

logger = get_logger(__name__)

#: Weight on lexical-marker density in the pre-saturation signal. Calibrated so a
#: text with roughly one dialect marker per ten tokens (density 0.1) scores about
#: 0.55, clearly dialectal without saturating on a single lexeme.
DEFAULT_MARKER_WEIGHT = 8.0

#: Weight on fired-rule density. Equal to the marker weight: a fired sound change
#: (psi <- pi) is about as diagnostic as a marker lexeme, and stacking both
#: pushes the score higher, as it should.
DEFAULT_FIRED_WEIGHT = 8.0


@TEXT_FEATURES.register("dialect_intensity")
class DialectIntensityExtractor(DenseTextExtractor):
    """A bounded dialectality index per document, plus a per-family breakdown.

    Emits ``intensity:overall`` and one ``intensity:<family>`` column per dialect
    family present in the lexicon or rule set (families sorted). Each cell is
    ``1 - exp(-signal)`` in ``[0, 1)``, where ``signal`` fuses that scope's
    marker density and detectable-rule fired density:

        ``signal = marker_weight * marker_density + fired_weight * fired_density``

    ``marker_density`` is the fraction of tokens matching a dialect marker (of the
    family, or of any family for ``overall``); ``fired_density`` is the summed
    per-token fired rate of the detectable rules (of the family, or all).
    Mergers contribute nothing here: their reflex is not positively
    identifiable, so intensity is built only from positive dialectal evidence.

    Args:
        lexicon_path: YAML marker lexicon replacing the bundled one (see
            :func:`tulip.features.text.keywords.load_lexicon`).
        rules_path: YAML rule file replacing the bundled one (see
            :func:`tulip.features.text.phonological_rules.load_phonological_rules`).
        marker_weight: Weight on marker density (see :data:`DEFAULT_MARKER_WEIGHT`).
        fired_weight: Weight on fired-rule density (see :data:`DEFAULT_FIRED_WEIGHT`).
    """

    def __init__(
        self,
        lexicon_path: str | Path | None = None,
        rules_path: str | Path | None = None,
        marker_weight: float = DEFAULT_MARKER_WEIGHT,
        fired_weight: float = DEFAULT_FIRED_WEIGHT,
    ) -> None:
        self.lexicon_path = lexicon_path
        self.rules_path = rules_path
        self.marker_weight = marker_weight
        self.fired_weight = fired_weight

    def fit(self, X: Sequence[str], y: Any = None) -> DialectIntensityExtractor:
        """Load the lexicon and rules and freeze the family column layout.

        Args:
            X: Ignored (the resources, not the data, define the features).
            y: Ignored (sklearn API compatibility).

        Returns:
            ``self``.

        Raises:
            ConfigurationError: If a weight is negative, or the lexicon/rules are
                missing or malformed.
        """
        if self.marker_weight < 0 or self.fired_weight < 0:
            raise ConfigurationError("marker_weight and fired_weight must be >= 0")

        lexicon = load_lexicon(self.lexicon_path)
        rules = load_phonological_rules(self.rules_path)

        #: All tokens that are a marker of *any* dialect (for the overall score).
        self.all_markers_: frozenset[str] = frozenset(
            marker for markers in lexicon.values() for marker in markers
        )
        #: family -> the union of its dialects' marker tokens.
        family_markers: dict[str, set[str]] = {}
        for key, markers in lexicon.items():
            family = family_for_lexicon_key(key)
            if family is not None:
                family_markers.setdefault(family, set()).update(markers)
        self.family_markers_: dict[str, frozenset[str]] = {
            family: frozenset(markers) for family, markers in family_markers.items()
        }
        #: family -> its detectable rewrite rules (mergers carry no fired signal).
        family_rules: dict[str, list[PhonologicalRule]] = {}
        self.detectable_rules_: tuple[PhonologicalRule, ...] = tuple(
            rule for rule in rules if rule.detectable
        )
        for rule in self.detectable_rules_:
            for family in rule.families:
                family_rules.setdefault(family, []).append(rule)
        self.family_rules_: dict[str, tuple[PhonologicalRule, ...]] = {
            family: tuple(rule_list) for family, rule_list in family_rules.items()
        }

        self.families_: tuple[str, ...] = tuple(
            sorted(set(self.family_markers_) | set(self.family_rules_))
        )
        self.feature_names_: tuple[str, ...] = (
            "intensity:overall",
            *(f"intensity:{family}" for family in self.families_),
        )
        logger.debug(
            "dialect_intensity: %d families, %d detectable rules",
            len(self.families_),
            len(self.detectable_rules_),
        )
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Compute the overall and per-family intensity for each document.

        Args:
            X: Sequence of raw text documents.

        Returns:
            Dense float64 array of shape ``(len(X), 1 + n_families)`` in
            ``[0, 1)``; all zeros for documents with no word tokens.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_fitted()
        documents = list(X)
        matrix = np.zeros((len(documents), len(self.feature_names_)), dtype=np.float64)
        for row, raw_text in enumerate(documents):
            tokens = word_tokens(str(raw_text), lowercase=True)
            if not tokens:
                continue
            matrix[row, 0] = self._intensity(
                self._marker_density(tokens, self.all_markers_),
                self._fired_density(tokens, self.detectable_rules_),
            )
            for column, family in enumerate(self.families_, start=1):
                matrix[row, column] = self._intensity(
                    self._marker_density(tokens, self.family_markers_.get(family, frozenset())),
                    self._fired_density(tokens, self.family_rules_.get(family, ())),
                )
        return matrix

    def _intensity(self, marker_density: float, fired_density: float) -> float:
        """Fuse the two densities and squash to ``[0, 1)`` (anchored at 0)."""
        signal = self.marker_weight * marker_density + self.fired_weight * fired_density
        return 1.0 - math.exp(-signal)

    @staticmethod
    def _marker_density(tokens: Sequence[str], markers: frozenset[str]) -> float:
        """Fraction of ``tokens`` that are one of ``markers``."""
        if not markers:
            return 0.0
        return sum(token in markers for token in tokens) / len(tokens)

    @staticmethod
    def _fired_density(tokens: Sequence[str], rules: Sequence[PhonologicalRule]) -> float:
        """Summed per-token fired rate over ``rules``."""
        return sum(rule.fired_rate(tokens) for rule in rules)
