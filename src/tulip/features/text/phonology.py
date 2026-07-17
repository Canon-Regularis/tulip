"""Phonological-pattern features for Polish dialect text.

Registers ``phonological_markers`` in
:data:`tulip.features.registries.TEXT_FEATURES`.

The shipped ``dialect_keywords`` lexicon matches whole words only, so it cannot
encode the sub-lexical sound-change markers that Polish dialectology treats as
primary diagnostics: *asynchronous soft labials* (Kurpie ``pi/bi/wi/mi`` ->
``psi/bzi/wzi/mni``) and *mazurzenie* (the Masovian merger ``cz/sz/ż/dż`` ->
``c/s/z/dz``). This module detects those patterns at the character level and
emits per-document rates the classifier can weigh.

Design:
    Each isogloss is a :class:`PhonologicalFeature`, a narrow ``Protocol`` with
    a single :meth:`~PhonologicalFeature.rate` method. The extractor holds
    a ``Sequence[PhonologicalFeature]`` and knows nothing about their kinds; it
    depends only on the Protocol. Concrete kinds are selected from YAML by
    a ``kind:`` key through the :data:`_FEATURE_KINDS` dict factory, so a third
    kind of isogloss is a new factory entry, not an edit to the extractor.

This is a feature extractor, not a predictor: it is a scikit-learn transformer
and is unrelated to the ``tulip.pipeline.protocols.SamplePredictor`` hierarchy.

Precision caveats to read before trusting either shipped feature:

* **Soft labials are only usable with an exclusion stoplist.** A bare
  ``psi|bzi|wzi|mni`` regex is noise: ``mni`` occurs in standard *mnie* ("me"),
  ``psi`` in standard *psi/psia/psie* ("canine"), ``wzi`` in standard *wziąć*
  ("to take"). :class:`IsoglossPattern` therefore skips any token on an
  ``exclude`` stoplist of the common standard collisions; that stoplist is what
  turns the raw pattern into a real signal.
* **Mazurzenie cannot be detected by a positive regex.** It *removes* the
  standard sibilant digraphs, replacing ``cz/sz/ż/dż`` with plain ``c/s/z/dz``,
  and standard Polish is full of legitimate ``c``, ``s``, ``z``. There is no
  positive string that marks a mazurzenie form. The honest operationalisation is
  the *absence* of the standard digraphs: a mazurzenie text has conspicuously
  few ``cz sz ż dż``. The shipped :class:`DigraphRate` feature ``sibilant_digraph``
  therefore measures the rate of those standard digraphs and lets the classifier
  learn that a LOW rate implies the Masovian merger. This module does not, and
  cannot, claim to positively "detect mazurzenie".

A starter isogloss set ships as package data
(``tulip/features/text/lexicons/isoglosses.yaml``); pass ``isogloss_path`` to
extend or replace it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from tulip.core.exceptions import ConfigurationError
from tulip.features.registries import TEXT_FEATURES
from tulip.features.text._base import DenseTextExtractor, check_per_tokens
from tulip.features.text._resource import read_versioned_entries
from tulip.features.text._tokenize import word_tokens
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

__all__ = [
    "DigraphRate",
    "IsoglossPattern",
    "PhonologicalFeature",
    "PhonologicalMarkerExtractor",
    "load_isoglosses",
]

logger = get_logger(__name__)

#: Name of the bundled starter isogloss set (package data under ``lexicons/``).
_BUNDLED_ISOGLOSSES = "isoglosses.yaml"

#: Only schema version this loader understands; bump on any breaking change.
_SCHEMA_VERSION = 1


@runtime_checkable
class PhonologicalFeature(Protocol):
    """One named, sub-lexical dialect signal reduced to a per-token rate.

    Implementations are frozen value objects selected from YAML by ``kind``.
    The extractor depends on this Protocol alone and treats every feature
    identically, so new kinds never touch the extractor.

    ``name`` is a read-only property, not a bare ``name: str`` attribute, so that
    the frozen-dataclass implementations (whose fields are read-only) satisfy it:
    a plain annotated attribute would demand a *settable* member they cannot
    provide. Access is identical either way (``feature.name``).

    Attributes:
        name: Stable identifier; becomes the ``phon:<name>`` output column.
    """

    @property
    def name(self) -> str:
        """Stable identifier; becomes the ``phon:<name>`` output column."""
        ...

    def rate(self, text: str, tokens: Sequence[str]) -> float:
        """Return the feature's occurrence rate as a fraction of ``tokens``.

        Args:
            text: The raw document (available to features that need character
                context beyond word tokens; the shipped kinds use ``tokens``).
            tokens: The document's word tokens, lowercased, as produced by
                :func:`tulip.features.text._tokenize.word_tokens`.

        Returns:
            Occurrences per token (``>= 0.0``); ``0.0`` when ``tokens`` is empty.
            The extractor scales this by its ``per_tokens`` base.
        """
        ...


@dataclass(frozen=True)
class IsoglossPattern:
    """Rate of a regex over attested dialectal spellings, minus a stoplist.

    A token counts when :attr:`pattern` matches inside it *and* the token is not
    on the :attr:`exclude` stoplist. The stoplist is essential, not optional:
    the dialectal clusters this kind targets (``psi/bzi/wzi/mni``) also occur in
    common standard words (*mnie*, *psie*, *wziąć*), and without excluding them
    the feature would fire on ordinary Polish.

    Attributes:
        name: Output-column stem (``phon:<name>``).
        pattern: Compiled regex matched with ``findall`` inside each token.
        exclude: Lowercased whole-token stoplist of standard collisions.
    """

    name: str
    pattern: re.Pattern[str]
    exclude: frozenset[str]

    def rate(self, text: str, tokens: Sequence[str]) -> float:
        """Matches per token, counting only tokens not on the stoplist."""
        if not tokens:
            return 0.0
        hits = sum(
            len(self.pattern.findall(token)) for token in tokens if token not in self.exclude
        )
        return hits / len(tokens)


@dataclass(frozen=True)
class DigraphRate:
    """Rate of a set of digraphs, per token.

    Used for the *absence*-based operationalisation of mazurzenie: measuring the
    rate of the standard sibilant digraphs ``cz/sz/ż/dż`` lets a classifier learn
    that a conspicuously LOW rate marks the Masovian merger. It does not detect
    mazurzenie positively: no positive string can, because the merger deletes
    the very digraphs it would key on.

    Attributes:
        name: Output-column stem (``phon:<name>``).
        digraphs: The digraph strings measured (lowercased), for introspection.
        matcher: Compiled alternation over :attr:`digraphs`, longest-first so
            e.g. ``dż`` is consumed before the bare ``ż``.
    """

    name: str
    digraphs: tuple[str, ...]
    matcher: re.Pattern[str]

    def rate(self, text: str, tokens: Sequence[str]) -> float:
        """Digraph occurrences per token across the document."""
        if not tokens:
            return 0.0
        hits = sum(len(self.matcher.findall(token)) for token in tokens)
        return hits / len(tokens)


def _coerce_word_list(raw: object, *, name: str, field: str, source: str) -> tuple[str, ...]:
    """Validate a YAML string list into a deduplicated lowercased tuple."""
    if not isinstance(raw, list):
        raise ConfigurationError(f"{source}: isogloss {name!r} field {field!r} must be a list")
    cleaned: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ConfigurationError(
                f"{source}: isogloss {name!r} field {field!r} has non-string entry {item!r}"
            )
        normalised = item.strip().lower()
        if normalised:
            cleaned.append(normalised)
    return tuple(dict.fromkeys(cleaned))


def _make_isogloss_pattern(
    *, name: str, entry: Mapping[str, Any], source: str
) -> PhonologicalFeature:
    """Build an :class:`IsoglossPattern` from a ``kind: pattern`` YAML entry."""
    pattern_src = entry.get("pattern")
    if not isinstance(pattern_src, str) or not pattern_src.strip():
        raise ConfigurationError(
            f"{source}: isogloss {name!r} (kind 'pattern') needs a non-empty 'pattern' regex"
        )
    try:
        compiled = re.compile(pattern_src)
    except re.error as exc:
        raise ConfigurationError(
            f"{source}: isogloss {name!r} has an invalid 'pattern' regex {pattern_src!r}: {exc}"
        ) from exc
    exclude = _coerce_word_list(entry.get("exclude", []), name=name, field="exclude", source=source)
    return IsoglossPattern(name=name, pattern=compiled, exclude=frozenset(exclude))


def _make_digraph_rate(*, name: str, entry: Mapping[str, Any], source: str) -> PhonologicalFeature:
    """Build a :class:`DigraphRate` from a ``kind: digraph`` YAML entry."""
    digraphs = _coerce_word_list(entry.get("digraphs"), name=name, field="digraphs", source=source)
    if not digraphs:
        raise ConfigurationError(
            f"{source}: isogloss {name!r} (kind 'digraph') needs a non-empty 'digraphs' list"
        )
    # Longest-first alternation so a multi-character digraph is consumed before
    # any single-character digraph nested inside it (e.g. ``dż`` before ``ż``).
    ordered = sorted(set(digraphs), key=len, reverse=True)
    matcher = re.compile("|".join(re.escape(digraph) for digraph in ordered))
    return DigraphRate(name=name, digraphs=digraphs, matcher=matcher)


#: Isogloss ``kind`` -> factory. Adding a kind is a new entry here plus its
#: value object; the extractor is never touched.
_FEATURE_KINDS: dict[str, Callable[..., PhonologicalFeature]] = {
    "pattern": _make_isogloss_pattern,
    "digraph": _make_digraph_rate,
}


def load_isoglosses(path: str | Path | None = None) -> tuple[PhonologicalFeature, ...]:
    """Load and validate a phonological isogloss set.

    Args:
        path: Path to a YAML isogloss file; ``None`` loads the bundled starter
            set. The file is a mapping with an integer ``version`` and a
            non-empty ``isoglosses`` list; each entry needs a unique ``name`` and
            a ``kind`` present in :data:`_FEATURE_KINDS`, plus that kind's fields
            (``pattern``/``exclude`` for ``pattern``, ``digraphs`` for
            ``digraph``).

    Returns:
        The isogloss features in file order.

    Raises:
        ConfigurationError: If the file is missing, is not a versioned mapping,
            has an empty/duplicate/malformed entry, names an unknown ``kind``, or
            carries an invalid regex or empty digraph list.
    """
    source, entries = read_versioned_entries(
        path,
        bundled_name=_BUNDLED_ISOGLOSSES,
        noun="isogloss",
        bundled_label="isoglosses",
        entity="isogloss",
        list_key="isoglosses",
        version=_SCHEMA_VERSION,
    )

    features: list[PhonologicalFeature] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigurationError(f"{source}: isogloss #{index} must be a mapping")
        raw_name = entry.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ConfigurationError(f"{source}: isogloss #{index} needs a non-empty 'name'")
        name = raw_name.strip()
        if name in seen:
            raise ConfigurationError(f"{source}: duplicate isogloss name {name!r}")
        seen.add(name)
        kind = entry.get("kind")
        factory = _FEATURE_KINDS.get(kind) if isinstance(kind, str) else None
        if factory is None:
            raise ConfigurationError(
                f"{source}: isogloss {name!r} has unknown kind {kind!r}; "
                f"known kinds: {', '.join(sorted(_FEATURE_KINDS))}"
            )
        features.append(factory(name=name, entry=entry, source=source))
    logger.debug("loaded %d isoglosses from %s", len(features), source)
    return tuple(features)


@TEXT_FEATURES.register("phonological_markers")
class PhonologicalMarkerExtractor(DenseTextExtractor):
    """Per-document rates of phonological dialect markers.

    Emits one ``phon:<name>`` column per isogloss (in file order). Each cell is
    the isogloss's per-token :meth:`PhonologicalFeature.rate` scaled by
    ``per_tokens``. Documents with no word tokens produce an all-zero row.

    ``per_tokens`` defaults to ``1.0``, a plain fraction of tokens, in [0, 1],
    rather than the "per 1000 tokens" convention that ``dialect_keywords`` uses,
    and the difference is deliberate. These are *dense* columns unioned with
    *sparse* TF-IDF blocks whose values also live in [0, 1]. An L2-penalised
    model regularises a column in inverse proportion to its magnitude, so a
    column scaled 1000x larger is effectively 1000x less regularised and drowns
    thousands of TF-IDF columns. Measured on the synthetic corpus, per-1000
    scaling turned a +0.02 accuracy gain into a -0.01 loss. Raise ``per_tokens``
    to 1000 when you want human-readable "hits per 1000 tokens" for reporting,
    not when feeding a linear model alongside TF-IDF.

    The extractor holds the isoglosses as an opaque ``Sequence[PhonologicalFeature]``
    and never inspects their concrete kind, so new isogloss kinds extend the
    output without changing this class.

    Args:
        isogloss_path: YAML isogloss file replacing the bundled starter set (see
            :func:`load_isoglosses`); ``None`` uses the bundled one.
        per_tokens: Normalisation base; rates are scaled to this many tokens.
            Leave at ``1.0`` when composing with TF-IDF (see above).
    """

    def __init__(
        self,
        isogloss_path: str | Path | None = None,
        per_tokens: float = 1.0,
    ) -> None:
        self.isogloss_path = isogloss_path
        self.per_tokens = per_tokens

    def fit(self, X: Sequence[str], y: Any = None) -> PhonologicalMarkerExtractor:
        """Load the isoglosses and freeze the column layout.

        Args:
            X: Ignored (the isogloss set, not the data, defines the features).
            y: Ignored (sklearn API compatibility).

        Returns:
            ``self``.

        Raises:
            ConfigurationError: If ``per_tokens`` is not positive or the isogloss
                set is missing/malformed.
        """
        check_per_tokens(self.per_tokens)
        self.features_: tuple[PhonologicalFeature, ...] = load_isoglosses(self.isogloss_path)
        self.feature_names_: tuple[str, ...] = tuple(
            f"phon:{feature.name}" for feature in self.features_
        )
        logger.debug("phonological_markers loaded %d isoglosses", len(self.features_))
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Compute scaled phonological-marker rates for each document.

        Args:
            X: Sequence of raw text documents.

        Returns:
            Dense float64 array of shape ``(len(X), n_isoglosses)``; all zeros
            for documents with no word tokens.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_fitted()
        documents = list(X)
        matrix = np.zeros((len(documents), len(self.feature_names_)), dtype=np.float64)
        for row, raw_text in enumerate(documents):
            text = str(raw_text)
            tokens = word_tokens(text, lowercase=True)
            if not tokens:
                continue
            for column, feature in enumerate(self.features_):
                matrix[row, column] = feature.rate(text, tokens) * self.per_tokens
        return matrix
