"""Contrastive dialect analysis: which linguistic features separate two dialects.

A leaderboard says *how well* a model tells dialects apart; it never says *what*
tells them apart. This answers the dialectology-native question directly: given
two dialect labels (for example ``silesian`` versus ``standard``), which
interpretable linguistic features are over-represented in one relative to the
other, in which direction, and by how much.

Three feature families are contrasted, the ones Polish dialectology treats as
primary and which the toolkit already operationalises:

* **lexical markers** from the marker lexicon
  (:func:`~tulip.features.text.keywords.load_lexicon`): dialect lexemes;
* **phonological isoglosses** from the rule engine
  (:func:`~tulip.features.text.phonological_rules.load_phonological_rules`): the
  sound changes (vowel and consonant) that fire in a sample;
* **morphological endings**: word-final suffixes, among the strongest written
  Polish dialect signals (mirroring
  :class:`~tulip.features.text.affixes.AffixFrequencyExtractor`).

For each feature the analysis compares its document-occurrence rate in the two
groups and reports a smoothed log-odds ratio (the effect size and its direction)
and a two-proportion z-test p-value, Holm-corrected across the whole comparison so
a long feature list does not manufacture significance. The evidence is
resource-defined and model-free: it reflects what the lexicon and rules find in
the gold-labelled text, not what any classifier learned, so it is a statement
about the *language*, not about a model. Every ordering is total and every float
is rounded, so a saved report is byte-stable.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import format_metric, markdown_table, save_report
from tulip._stats import holm_correct, two_proportion_p
from tulip.core.exceptions import ConfigurationError
from tulip.labels.taxonomy import LabelLevel, display_name

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tulip.core.types import Sample
    from tulip.features.text.phonological_rules import PhonologicalRule

__all__ = ["ContrastFeature", "ContrastReport", "contrast_dialects"]

#: Stored floats are rounded to this many digits so a saved report is byte-stable.
CONTRAST_FLOAT_DIGITS = 6

#: Word-final suffix length used for the morphological-ending feature.
DEFAULT_SUFFIX_LEN = 3

#: A feature must appear in at least this many documents (across both groups) to
#: be contrasted; rarer features are too noisy to rank.
DEFAULT_MIN_SUPPORT = 5

#: The three feature families, in report order.
_CATEGORIES = ("lexical", "phonological", "morphological")
_CATEGORY_TITLES = {
    "lexical": "Lexical markers",
    "phonological": "Phonological isoglosses",
    "morphological": "Morphological endings",
}


class ContrastFeature(BaseModel):
    """One linguistic feature and how strongly it separates the two dialects."""

    model_config = ConfigDict(frozen=True)

    category: str
    feature: str
    n_a: int = Field(ge=0)
    n_b: int = Field(ge=0)
    rate_a: float = Field(ge=0.0, le=1.0)
    rate_b: float = Field(ge=0.0, le=1.0)
    log_odds: float
    favored: str
    p_value: float = Field(ge=0.0, le=1.0)
    p_value_holm: float = Field(ge=0.0, le=1.0)

    @property
    def significant(self) -> bool:
        """Whether the Holm-corrected contrast clears the conventional 0.05 cutoff."""
        return self.p_value_holm < 0.05


class ContrastReport(BaseModel):
    """The features that most distinguish dialect ``a`` from dialect ``b``."""

    model_config = ConfigDict(frozen=True)

    level: str
    dialect_a: str
    dialect_b: str
    n_docs_a: int = Field(ge=0)
    n_docs_b: int = Field(ge=0)
    features: tuple[ContrastFeature, ...]

    def to_markdown(self, *, top_k: int = 10) -> str:
        """Render the contrast per feature family, each direction shown separately."""
        title = (
            f"# Contrastive analysis: {display_name(self.dialect_a)} vs "
            f"{display_name(self.dialect_b)} (level={self.level})"
        )
        note = (
            f"{self.n_docs_a} vs {self.n_docs_b} documents. Log-odds is the smoothed "
            f"ratio of a feature's document-occurrence odds; a positive value favours "
            f"{display_name(self.dialect_a)}, negative favours "
            f"{display_name(self.dialect_b)}. p is a two-proportion z-test, "
            f"Holm-corrected; * marks p < 0.05. Evidence is resource-defined and "
            f"model-free."
        )
        parts = [title, note]
        for category in _CATEGORIES:
            here = [feature for feature in self.features if feature.category == category]
            parts.append(f"## {_CATEGORY_TITLES[category]}")
            if not here:
                parts.append("No features cleared the support threshold.")
                continue
            for dialect in (self.dialect_a, self.dialect_b):
                favouring = [feature for feature in here if feature.favored == dialect][:top_k]
                parts.append(f"### Favouring {display_name(dialect)}")
                parts.append(_feature_table(favouring, self.dialect_a, self.dialect_b))
        return "\n\n".join(parts)

    def save(self, path: Path | str) -> None:
        """Write the report as deterministic JSON (sorted keys, rounded floats)."""
        save_report(self, path, digits=CONTRAST_FLOAT_DIGITS)


def contrast_dialects(
    samples: Sequence[Sample],
    dialect_a: str,
    dialect_b: str,
    *,
    level: LabelLevel = LabelLevel.DIALECT,
    lexicon_path: str | Path | None = None,
    rules_path: str | Path | None = None,
    suffix_len: int = DEFAULT_SUFFIX_LEN,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> ContrastReport:
    """Contrast the linguistic features of two dialects over gold-labelled text.

    Args:
        samples: Labelled samples carrying text; those labelled ``dialect_a`` or
            ``dialect_b`` at ``level`` are contrasted, the rest ignored.
        dialect_a: The first dialect label (positive log-odds favours it).
        dialect_b: The second dialect label.
        level: Taxonomy level the labels are read at (e.g. ``family`` to contrast
            ``silesian`` against ``standard``).
        lexicon_path: Marker lexicon file replacing the bundled one.
        rules_path: Phonological rule file replacing the bundled one.
        suffix_len: Word-final length used for the morphological-ending feature.
        min_support: Minimum documents (across both groups) a feature must occur in
            to be contrasted.

    Returns:
        A :class:`ContrastReport`, features sorted by descending effect size within
        each family and direction.

    Raises:
        ConfigurationError: if the two labels are equal, ``suffix_len`` or
            ``min_support`` is not positive, or either dialect has no text samples.
    """
    if dialect_a == dialect_b:
        raise ConfigurationError(f"cannot contrast {dialect_a!r} with itself")
    if suffix_len < 1:
        raise ConfigurationError(f"suffix_len must be >= 1, got {suffix_len}")
    if min_support < 1:
        raise ConfigurationError(f"min_support must be >= 1, got {min_support}")

    from tulip.features.text.keywords import load_lexicon
    from tulip.features.text.phonological_rules import load_phonological_rules

    markers = frozenset(marker for group in load_lexicon(lexicon_path).values() for marker in group)
    rules = tuple(rule for rule in load_phonological_rules(rules_path) if rule.detectable)

    docs_a = _texts_for(samples, level, dialect_a)
    docs_b = _texts_for(samples, level, dialect_b)
    if not docs_a:
        raise ConfigurationError(f"no text samples labelled {dialect_a!r} at level {level.value}")
    if not docs_b:
        raise ConfigurationError(f"no text samples labelled {dialect_b!r} at level {level.value}")

    present_a = _feature_document_counts(
        docs_a, markers=markers, rules=rules, suffix_len=suffix_len
    )
    present_b = _feature_document_counts(
        docs_b, markers=markers, rules=rules, suffix_len=suffix_len
    )
    n_a, n_b = len(docs_a), len(docs_b)

    scored: list[tuple[float, str, str, int, int]] = []
    for feature in present_a.keys() | present_b.keys():
        count_a, count_b = present_a[feature], present_b[feature]
        if count_a + count_b < min_support:
            continue
        p_value = two_proportion_p(count_a, n_a, count_b, n_b)
        category, name = feature
        scored.append((p_value, category, name, count_a, count_b))

    holm = holm_correct([item[0] for item in scored])
    features = [
        _build_feature(category, name, count_a, count_b, n_a, n_b, dialect_a, dialect_b, p, p_holm)
        for (p, category, name, count_a, count_b), p_holm in zip(scored, holm, strict=True)
    ]
    # Effect size first (largest separation), then p-value, then name for a total
    # order; grouping by category/direction happens at render time.
    features.sort(
        key=lambda feature: (-abs(feature.log_odds), feature.p_value_holm, feature.feature)
    )
    return ContrastReport(
        level=level.value,
        dialect_a=dialect_a,
        dialect_b=dialect_b,
        n_docs_a=n_a,
        n_docs_b=n_b,
        features=tuple(features),
    )


def _texts_for(samples: Sequence[Sample], level: LabelLevel, label: str) -> list[str]:
    """Texts of samples labelled ``label`` at ``level`` (audio-only samples skipped)."""
    return [
        str(sample.text)
        for sample in samples
        if sample.text is not None and sample.labels.at_level(level) == label
    ]


def _feature_document_counts(
    texts: Sequence[str],
    *,
    markers: frozenset[str],
    rules: tuple[PhonologicalRule, ...],
    suffix_len: int,
) -> Counter[tuple[str, str]]:
    """Count, per feature, how many documents contain it (binary presence)."""
    counts: Counter[tuple[str, str]] = Counter()
    for text in texts:
        counts.update(_document_features(text, markers=markers, rules=rules, suffix_len=suffix_len))
    return counts


def _document_features(
    text: str,
    *,
    markers: frozenset[str],
    rules: tuple[PhonologicalRule, ...],
    suffix_len: int,
) -> set[tuple[str, str]]:
    """The distinct linguistic features present in one document."""
    from tulip.features.text._tokenize import word_tokens

    tokens = word_tokens(text, lowercase=True)
    token_set = set(tokens)
    features: set[tuple[str, str]] = set()
    features.update(("lexical", marker) for marker in markers & token_set)
    for rule in rules:
        if any(rule.fired_matches(token) for token in tokens):
            features.add(("phonological", rule.name))
    for token in tokens:
        if len(token) > suffix_len:
            features.add(("morphological", f"-{token[-suffix_len:]}"))
    return features


def _build_feature(
    category: str,
    name: str,
    count_a: int,
    count_b: int,
    n_a: int,
    n_b: int,
    dialect_a: str,
    dialect_b: str,
    p_value: float,
    p_value_holm: float,
) -> ContrastFeature:
    """Assemble a :class:`ContrastFeature` with its smoothed log-odds ratio."""
    log_odds = _log_odds_ratio(count_a, n_a, count_b, n_b)
    return ContrastFeature(
        category=category,
        feature=name,
        n_a=count_a,
        n_b=count_b,
        rate_a=count_a / n_a,
        rate_b=count_b / n_b,
        log_odds=log_odds,
        favored=dialect_a if log_odds >= 0 else dialect_b,
        p_value=p_value,
        p_value_holm=p_value_holm,
    )


def _log_odds_ratio(count_a: int, n_a: int, count_b: int, n_b: int) -> float:
    """Haldane-Anscombe (0.5-smoothed) log-odds ratio of presence in a versus b."""
    a_present, a_absent = count_a + 0.5, n_a - count_a + 0.5
    b_present, b_absent = count_b + 0.5, n_b - count_b + 0.5
    return math.log((a_present * b_absent) / (a_absent * b_present))


def _feature_table(features: Sequence[ContrastFeature], dialect_a: str, dialect_b: str) -> str:
    """Render a table of contrast features (rates, log-odds, Holm p)."""
    headers = (
        "Feature",
        f"Rate {display_name(dialect_a)}",
        f"Rate {display_name(dialect_b)}",
        "Log-odds",
        "p (Holm)",
    )
    rows = [
        (
            feature.feature + (" *" if feature.significant else ""),
            format_metric(feature.rate_a),
            format_metric(feature.rate_b),
            format_metric(feature.log_odds),
            format_metric(feature.p_value_holm),
        )
        for feature in features
    ] or [("n/a", "n/a", "n/a", "n/a", "n/a")]
    return markdown_table(headers, rows)
