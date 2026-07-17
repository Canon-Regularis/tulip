"""Bidirectional phonological rule engine for Polish dialect text.

Registers ``phonological_rules`` in
:data:`tulip.features.registries.TEXT_FEATURES`.

Where :mod:`tulip.features.text.phonology` measures the *rate* of a few
isoglosses, this module models the group-defining sound changes as ordered,
reversible **rewrite rules** and does three things a rate detector cannot:

* **Distinguish environment from change.** Every rule reports both an
  ``applicable`` rate (the standard environment where the change *could* fire is
  present) and, when the reflex is positively identifiable, a ``fired`` rate (the
  dialectal reflex *is* present). "environment present, change absent" is
  standard Polish; "change fired" is dialectal, a distinction a single rate
  column blurs.
* **Normalise dialect -> standard.** Running the detectable rules in reverse
  (:func:`normalize_to_standard`) collapses dialectal spellings back to their
  standard form, so cross-corpus matching and deduplication see one lexeme where
  the surface has several.
* **Stay honest about mergers.** A merger (mazurzenie's ``cz -> c``, kaszubienie's
  ``ś -> s``) produces a reflex indistinguishable from ordinary standard Polish,
  so it cannot be positively detected and cannot be reversed. Such a rule is
  marked ``detectable: false``: it contributes only an ``applicable`` rate (the
  standard forms, whose absence is the signal) and its reverse is a no-op.

One :class:`PhonologicalRule` value object covers both cases, parameterised by
``detectable`` and the token position it applies at, so a new isogloss is a new
YAML entry (``lexicons/phonological_rules.yaml``), never a code change. The
extractor holds an opaque ``Sequence[PhonologicalRule]`` and depends on nothing
concrete, mirroring :mod:`tulip.features.text.phonology`.

Detection runs on lowercased letter-only word tokens (:func:`word_tokens`), so
this stays consistent with the other lexical features and with ``isoglosses.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tulip.core.exceptions import ConfigurationError
from tulip.features.registries import TEXT_FEATURES
from tulip.features.text._base import DenseTextExtractor, check_per_tokens
from tulip.features.text._resource import read_versioned_entries
from tulip.features.text._tokenize import word_tokens
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = [
    "PhonologicalRule",
    "PhonologicalRuleExtractor",
    "apply_rules",
    "load_phonological_rules",
    "normalize_to_standard",
]

logger = get_logger(__name__)

#: Name of the bundled rule set (package data under ``lexicons/``).
_BUNDLED_RULES = "phonological_rules.yaml"

#: Only schema version this loader understands; bump on any breaking change.
_SCHEMA_VERSION = 1

#: Token positions a rule may apply at, mapped to (regex prefix, regex suffix).
_ANCHORS: dict[str, tuple[str, str]] = {
    "anywhere": ("", ""),
    "initial": ("^", ""),
    "final": ("", "$"),
}


@dataclass(frozen=True)
class PhonologicalRule:
    """One ordered, position-aware phonological rewrite.

    A rule rewrites standard graphemes to their dialectal reflex (:attr:`pairs`,
    standard -> dialectal, longest-source-first). It applies only inside tokens
    at :attr:`where`, and never on tokens listed in :attr:`exclude` (the standard
    words whose letters collide with the rule's clusters).

    ``detectable`` splits the two dialectological cases: a positively
    identifiable change (soft labials) exposes a ``fired`` rate and a working
    reverse; a merger (mazurzenie) does neither: its reflex is ordinary
    standard Polish, so only its ``applicable`` rate carries signal.

    The compiled matchers and the forward/reverse lookup maps are built once by
    :func:`load_phonological_rules`; the value object is otherwise pure data.
    """

    name: str
    families: tuple[str, ...]
    dialects: tuple[str, ...]
    detectable: bool
    where: str
    pairs: tuple[tuple[str, str], ...]
    exclude: frozenset[str]
    description: str
    attestation: str
    _forward: Mapping[str, str]
    _reverse: Mapping[str, str]
    _standard_matcher: re.Pattern[str]
    _dialectal_matcher: re.Pattern[str]

    def applicable_rate(self, tokens: Sequence[str]) -> float:
        """Standard-environment occurrences per token (``0.0`` when no tokens)."""
        if not tokens:
            return 0.0
        return sum(len(self.applicable_matches(token)) for token in tokens) / len(tokens)

    def fired_rate(self, tokens: Sequence[str]) -> float:
        """Dialectal-reflex occurrences per token; always ``0.0`` for a merger."""
        if not tokens:
            return 0.0
        return sum(len(self.fired_matches(token)) for token in tokens) / len(tokens)

    def applicable_matches(self, token: str) -> list[str]:
        """The standard-environment substrings this rule matches in ``token``."""
        if token in self.exclude:
            return []
        return self._standard_matcher.findall(token)

    def fired_matches(self, token: str) -> list[str]:
        """The dialectal-reflex substrings in ``token`` (empty for a merger)."""
        if not self.detectable or token in self.exclude:
            return []
        return self._dialectal_matcher.findall(token)

    def apply_token(self, token: str) -> str:
        """Rewrite one token standard -> dialectal (identity if excluded)."""
        return self._rewrite(token, self._standard_matcher, self._forward)

    def normalize_token(self, token: str) -> str:
        """Rewrite one token dialectal -> standard; identity for a merger."""
        if not self.detectable:
            return token
        return self._rewrite(token, self._dialectal_matcher, self._reverse)

    def _rewrite(self, token: str, matcher: re.Pattern[str], mapping: Mapping[str, str]) -> str:
        if token in self.exclude:
            return token
        return matcher.sub(lambda match: mapping[match.group()], token)


def load_phonological_rules(path: str | Path | None = None) -> tuple[PhonologicalRule, ...]:
    """Load and validate the phonological rule set.

    Args:
        path: YAML rule file; ``None`` loads the bundled set. The file is a
            mapping with an integer ``version`` and a non-empty ``rules`` list;
            each entry needs a unique ``name``, a ``map`` of standard -> dialectal
            substitutions, a ``where`` in ``{anywhere, initial, final}``, and a
            boolean ``detectable``; ``families``/``dialects``/``exclude`` are
            optional.

    Returns:
        The rules in file order, each with its matchers compiled.

    Raises:
        ConfigurationError: if the file is missing, is not a versioned mapping,
            has an empty/duplicate/malformed entry, or names an unknown ``where``.
    """
    source, entries = read_versioned_entries(
        path,
        bundled_name=_BUNDLED_RULES,
        noun="phonological rules",
        bundled_label="rules",
        entity="rule",
        list_key="rules",
        version=_SCHEMA_VERSION,
    )

    rules: list[PhonologicalRule] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigurationError(f"{source}: rule #{index} must be a mapping")
        rule = _build_rule(entry, index=index, source=source)
        if rule.name in seen:
            raise ConfigurationError(f"{source}: duplicate rule name {rule.name!r}")
        seen.add(rule.name)
        rules.append(rule)
    logger.debug("loaded %d phonological rules from %s", len(rules), source)
    return tuple(rules)


def normalize_to_standard(text: str, *, rules: Sequence[PhonologicalRule] | None = None) -> str:
    """Collapse dialectal spellings back towards standard Polish.

    Applies every *detectable* rule's reverse to each lowercased word token and
    rejoins them with single spaces. This is a canonicalisation for matching and
    deduplication (it lowercases and drops punctuation), not a display transform;
    mergers are intentionally left untouched because they cannot be reversed.

    Args:
        text: The document to normalise.
        rules: Rule set to use; ``None`` loads the bundled set.

    Returns:
        The normalised token stream (lowercased, space-joined).
    """
    active = tuple(rules) if rules is not None else load_phonological_rules()
    normalised = []
    for token in word_tokens(text, lowercase=True):
        for rule in active:
            token = rule.normalize_token(token)
        normalised.append(token)
    return " ".join(normalised)


def apply_rules(text: str, *, rules: Sequence[PhonologicalRule] | None = None) -> str:
    """Apply every rule forward (standard -> dialectal) to each word token.

    The inverse of :func:`normalize_to_standard`, used mainly to check that the
    engine reproduces the corpus generator's transforms. Operates on lowercased
    word tokens and rejoins with single spaces.

    Args:
        text: The document to transform.
        rules: Rule set to use; ``None`` loads the bundled set.

    Returns:
        The transformed token stream (lowercased, space-joined).
    """
    active = tuple(rules) if rules is not None else load_phonological_rules()
    transformed = []
    for token in word_tokens(text, lowercase=True):
        for rule in active:
            token = rule.apply_token(token)
        transformed.append(token)
    return " ".join(transformed)


def _build_rule(entry: Mapping[str, Any], *, index: int, source: str) -> PhonologicalRule:
    """Construct a :class:`PhonologicalRule` from one validated YAML entry."""
    name = _require_name(entry, index=index, source=source)
    where = entry.get("where", "anywhere")
    if where not in _ANCHORS:
        raise ConfigurationError(
            f"{source}: rule {name!r} has unknown where {where!r}; "
            f"expected one of {', '.join(sorted(_ANCHORS))}"
        )
    detectable = entry.get("detectable", True)
    if not isinstance(detectable, bool):
        raise ConfigurationError(f"{source}: rule {name!r} field 'detectable' must be a boolean")
    pairs = _coerce_map(entry.get("map"), name=name, source=source)
    exclude = _coerce_exclude(entry.get("exclude", []), name=name, source=source)
    prefix, suffix = _ANCHORS[where]
    forward = dict(pairs)
    reverse = {dialectal: standard for standard, dialectal in pairs}
    return PhonologicalRule(
        name=name,
        families=_coerce_str_tuple(entry.get("families", [])),
        dialects=_coerce_str_tuple(entry.get("dialects", [])),
        detectable=detectable,
        where=where,
        pairs=pairs,
        exclude=exclude,
        description=str(entry.get("description", "")).strip(),
        attestation=str(entry.get("source", "")).strip(),
        _forward=forward,
        _reverse=reverse,
        _standard_matcher=_alternation(forward, prefix, suffix),
        _dialectal_matcher=_alternation(reverse, prefix, suffix),
    )


def _alternation(mapping: Mapping[str, str], prefix: str, suffix: str) -> re.Pattern[str]:
    """Compile a longest-first alternation over ``mapping``'s keys, anchored."""
    ordered = sorted(mapping, key=len, reverse=True)
    body = "|".join(re.escape(key) for key in ordered)
    return re.compile(f"{prefix}(?:{body}){suffix}")


def _require_name(entry: Mapping[str, Any], *, index: int, source: str) -> str:
    raw = entry.get("name")
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError(f"{source}: rule #{index} needs a non-empty 'name'")
    return raw.strip()


def _coerce_map(raw: Any, *, name: str, source: str) -> tuple[tuple[str, str], ...]:
    """Validate a ``map`` into ordered, lowercased (standard, dialectal) pairs."""
    if not isinstance(raw, dict) or not raw:
        raise ConfigurationError(f"{source}: rule {name!r} needs a non-empty 'map' mapping")
    pairs: list[tuple[str, str]] = []
    for standard, dialectal in raw.items():
        if not isinstance(standard, str) or not isinstance(dialectal, str):
            raise ConfigurationError(
                f"{source}: rule {name!r} 'map' entries must be string -> string"
            )
        left, right = standard.strip().lower(), dialectal.strip().lower()
        if not left or not right:
            raise ConfigurationError(f"{source}: rule {name!r} has an empty 'map' key or value")
        pairs.append((left, right))
    # Longest source first so a multi-character cluster is consumed before any
    # shorter one nested inside it (e.g. ``dż`` before the bare ``ż``).
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return tuple(pairs)


def _coerce_exclude(raw: Any, *, name: str, source: str) -> frozenset[str]:
    """Validate an ``exclude`` stoplist into a lowercased frozenset."""
    if not isinstance(raw, list):
        raise ConfigurationError(f"{source}: rule {name!r} field 'exclude' must be a list")
    cleaned = [item.strip().lower() for item in raw if isinstance(item, str) and item.strip()]
    return frozenset(cleaned)


def _coerce_str_tuple(raw: Any) -> tuple[str, ...]:
    """Validate an optional string list into a deduplicated lowercased tuple."""
    if not isinstance(raw, list):
        return ()
    cleaned = [item.strip().lower() for item in raw if isinstance(item, str) and item.strip()]
    return tuple(dict.fromkeys(cleaned))


@TEXT_FEATURES.register("phonological_rules")
class PhonologicalRuleExtractor(DenseTextExtractor):
    """Per-document applicable/fired rates for the phonological rewrite rules.

    Emits ``rule:<name>:applicable`` for every rule and, additionally,
    ``rule:<name>:fired`` for every *detectable* rule. Each cell is the rule's
    per-token rate scaled by ``per_tokens``. Together the two columns let a model
    separate "the standard environment is present but unchanged" (standard) from
    "the dialectal reflex fired" (dialectal), the distinction a single rate
    cannot express.

    ``per_tokens`` defaults to ``1.0`` (a plain fraction of tokens) for the same
    reason as :class:`~tulip.features.text.phonology.PhonologicalMarkerExtractor`:
    these dense columns are unioned with TF-IDF blocks in [0, 1], and a larger
    scale would be under-regularised by an L2 model and drown the sparse columns.

    Args:
        rules_path: YAML rule file replacing the bundled set (see
            :func:`load_phonological_rules`); ``None`` uses the bundled one.
        per_tokens: Normalisation base; rates are scaled to this many tokens.
    """

    def __init__(
        self,
        rules_path: str | Path | None = None,
        per_tokens: float = 1.0,
    ) -> None:
        self.rules_path = rules_path
        self.per_tokens = per_tokens

    def fit(self, X: Sequence[str], y: Any = None) -> PhonologicalRuleExtractor:
        """Load the rules and freeze the column layout.

        Args:
            X: Ignored (the rule set, not the data, defines the features).
            y: Ignored (sklearn API compatibility).

        Returns:
            ``self``.

        Raises:
            ConfigurationError: If ``per_tokens`` is not positive or the rule set
                is missing/malformed.
        """
        check_per_tokens(self.per_tokens)
        self.rules_: tuple[PhonologicalRule, ...] = load_phonological_rules(self.rules_path)
        names: list[str] = []
        for rule in self.rules_:
            names.append(f"rule:{rule.name}:applicable")
            if rule.detectable:
                names.append(f"rule:{rule.name}:fired")
        self.feature_names_: tuple[str, ...] = tuple(names)
        logger.debug(
            "phonological_rules loaded %d rules -> %d columns", len(self.rules_), len(names)
        )
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Compute scaled applicable/fired rates for each document.

        Args:
            X: Sequence of raw text documents.

        Returns:
            Dense float64 array of shape ``(len(X), n_columns)``; all zeros for
            documents with no word tokens.

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
            column = 0
            for rule in self.rules_:
                matrix[row, column] = rule.applicable_rate(tokens) * self.per_tokens
                column += 1
                if rule.detectable:
                    matrix[row, column] = rule.fired_rate(tokens) * self.per_tokens
                    column += 1
        return matrix
