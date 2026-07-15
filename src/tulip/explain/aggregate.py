"""Corpus-level dialect-evidence roll-up: which phenomena separate the dialects.

The :class:`~tulip.explain.dialect_evidence.DialectEvidenceExplainer` names, for a
single text, the marker lexemes it matched and the isoglosses that fired. This
module sums those per-sample findings across a whole labelled corpus into one
summary a dialectologist can validate: for each phenomenon, how often it occurs,
which gold dialect its carriers belong to, and how concentrated that association
is.

The concentration measure is class-conditional lift: the probability that a
carrier of a phenomenon has gold class ``c``, divided by the base rate of ``c``.
A lift above one means the phenomenon is over-represented in that class, so a
high-lift isogloss is one that genuinely separates a dialect rather than
appearing everywhere. Phenomena carried by too few samples are flagged
low-support and never headline, because a lift computed from one or two carriers
is noise.

Two design choices keep this honest and reproducible.

The evidence is resource-defined, so the roll-up needs no fitted model. The
markers and rules are found in the text by the lexicon and the isogloss rules,
not inferred by a classifier. Running the aggregation over the raw resources
therefore validates the linguistic resources against the gold taxonomy directly,
and makes the report identical no matter which model is in play. The explainer is
invoked with no pipeline, so it reports evidence without a predicted label.

The lift axis is the gold label, not a prediction. This asks whether the
resources line up with the truth, which is the dialectological question, and it
keeps the report deterministic and byte-stable: the same corpus always yields the
same bytes, and every stored float is rounded to a fixed precision.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import write_sorted_json
from tulip.explain.dialect_evidence import DialectEvidenceExplainer
from tulip.labels.taxonomy import LabelLevel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tulip.core.types import Sample

__all__ = [
    "ClassCount",
    "FamilyEvidence",
    "GlobalEvidenceReport",
    "PhenomenonFrequency",
    "dataset_evidence",
]

#: Phenomena carried by fewer than this many samples are flagged low-support:
#: their class association is real but too sparse to read as a headline finding.
DEFAULT_MIN_SUPPORT = 3

#: Stored floats (lift, share) are rounded to this many digits, so a saved report
#: is byte-identical when the content is, mirroring the other rigor reports.
EVIDENCE_FLOAT_DIGITS = 6

#: The evidence groups carried in an ``Explanation.details`` payload.
_EVIDENCE_GROUPS = ("markers", "fired_rules", "applicable_rules")


class ClassCount(BaseModel):
    """How many carriers of a phenomenon fall in one gold class."""

    model_config = ConfigDict(frozen=True)

    label: str
    count: int = Field(ge=1)


class PhenomenonFrequency(BaseModel):
    """One dialectal phenomenon, tallied and scored across the corpus."""

    model_config = ConfigDict(frozen=True)

    phenomenon: str
    label: str
    families: tuple[str, ...]
    n_samples: int = Field(ge=1)
    total_count: int = Field(ge=1)
    top_class: str
    top_class_lift: float = Field(ge=0.0)
    top_class_share: float = Field(ge=0.0, le=1.0)
    low_support: bool
    class_counts: tuple[ClassCount, ...]


class FamilyEvidence(BaseModel):
    """Corpus-wide positive dialectal evidence attributed to one family."""

    model_config = ConfigDict(frozen=True)

    family: str
    n_samples: int = Field(ge=1)
    total_count: int = Field(ge=1)


class GlobalEvidenceReport(BaseModel):
    """The dialectal evidence found across a labelled corpus, ranked by lift."""

    model_config = ConfigDict(frozen=True)

    name: str
    level: str
    n_samples: int = Field(ge=0)
    n_skipped: int = Field(ge=0)
    phenomena: tuple[PhenomenonFrequency, ...]
    families: tuple[FamilyEvidence, ...]

    @property
    def most_diagnostic(self) -> PhenomenonFrequency | None:
        """The highest-lift phenomenon with adequate support, or ``None``.

        Phenomena are stored reliable-first, so this is the top reliable entry.
        Returns ``None`` when every phenomenon is low-support or none was found.
        """
        for phenomenon in self.phenomena:
            if not phenomenon.low_support:
                return phenomenon
        return None

    def to_markdown(self, *, top_k: int = 20) -> str:
        """Render the roll-up as markdown, most diagnostic phenomenon first."""
        from tulip.evaluation._format import format_metric, markdown_table

        headline = self.most_diagnostic
        if headline is None:
            summary = "Most diagnostic phenomenon: none with adequate support"
        else:
            summary = (
                f"Most diagnostic phenomenon: {headline.label} ({headline.phenomenon}) "
                f"-> {headline.top_class} (lift {format_metric(headline.top_class_lift)})"
            )
        note = (
            "Evidence is resource-defined (marker lexicon and isogloss rules); lift is over "
            "gold labels. Low-support phenomena are marked *."
        )
        phenomenon_rows = [
            (
                phenomenon.phenomenon,
                phenomenon.label + (" *" if phenomenon.low_support else ""),
                ", ".join(phenomenon.families) or "n/a",
                str(phenomenon.n_samples),
                str(phenomenon.total_count),
                phenomenon.top_class,
                format_metric(phenomenon.top_class_lift),
                format_metric(phenomenon.top_class_share),
            )
            for phenomenon in self.phenomena[:top_k]
        ]
        family_rows = [
            (family.family, str(family.n_samples), str(family.total_count))
            for family in self.families
        ]
        parts = [
            f"# Dialect evidence: {self.name} ({self.n_samples} labelled, level={self.level})",
            summary,
            note,
            "## Phenomena by class-conditional lift",
            markdown_table(
                (
                    "Phenomenon",
                    "Label",
                    "Families",
                    "Samples",
                    "Occurrences",
                    "Top class",
                    "Lift",
                    "Share",
                ),
                phenomenon_rows or [("n/a", "n/a", "n/a", "0", "0", "n/a", "n/a", "n/a")],
            ),
            "## Family evidence",
            markdown_table(
                ("Family", "Samples", "Occurrences"),
                family_rows or [("n/a", "0", "0")],
            ),
        ]
        return "\n\n".join(parts)

    def save(self, path: Path | str) -> None:
        """Write the report as deterministic JSON (sorted keys)."""
        write_sorted_json(Path(path), self.model_dump(mode="json"))


class _Tally:
    """Mutable per-phenomenon accumulator, keyed by (phenomenon type, label)."""

    __slots__ = ("carriers", "class_carriers", "families", "total")

    def __init__(self, families: tuple[str, ...]) -> None:
        self.families = families
        self.total = 0
        self.carriers = 0
        self.class_carriers: dict[str, int] = {}


def dataset_evidence(
    samples: Iterable[Sample],
    *,
    level: LabelLevel = LabelLevel.DIALECT,
    name: str = "dataset",
    min_support: int = DEFAULT_MIN_SUPPORT,
    explainer: DialectEvidenceExplainer | None = None,
) -> GlobalEvidenceReport:
    """Roll per-sample dialectal evidence up into a corpus-level summary.

    Each labelled text is passed through the dialect-evidence explainer; the
    matched markers and fired isoglosses are tallied, and each phenomenon is
    scored by the class-conditional lift of its most associated gold class.

    Args:
        samples: The labelled corpus. A sample with no usable text (missing,
            blank, or whitespace-only), or no gold label at ``level``, is skipped
            and counted in ``n_skipped``.
        level: Gold label granularity for the lift axis (dialect by default).
        name: Corpus name recorded in the report header.
        min_support: Carrier count below which a phenomenon is flagged
            low-support and kept out of the headline.
        explainer: An explainer to reuse; a default one is built when omitted.

    Returns:
        A :class:`GlobalEvidenceReport` with phenomena ranked reliable-first then
        by descending lift, and a corpus-wide per-family evidence tally.
    """
    explainer = explainer or DialectEvidenceExplainer()

    tallies: dict[tuple[str, str], _Tally] = {}
    class_size: dict[str, int] = {}
    family_docs: dict[str, int] = {}
    family_total: dict[str, int] = {}
    n_analysed = 0
    n_skipped = 0

    for sample in samples:
        text = sample.text
        gold = sample.labels.at_level(level)
        # Skip a row the explainer cannot use: no text, blank or whitespace-only
        # text (which the explainer rejects), or no gold label at this level.
        if not text or not text.strip() or gold is None:
            n_skipped += 1
            continue
        n_analysed += 1
        class_size[gold] = class_size.get(gold, 0) + 1
        details = explainer.explain(None, text).details
        _accumulate_phenomena(details, tallies, gold)
        _accumulate_families(details, family_docs, family_total)

    phenomena = tuple(
        sorted(
            (
                _phenomenon_frequency(key, tally, class_size, n_analysed, min_support)
                for key, tally in tallies.items()
            ),
            key=_rank_key,
        )
    )
    families = tuple(
        FamilyEvidence(
            family=family, n_samples=family_docs[family], total_count=family_total[family]
        )
        for family in sorted(family_docs)
    )
    return GlobalEvidenceReport(
        name=name,
        level=level.value,
        n_samples=n_analysed,
        n_skipped=n_skipped,
        phenomena=phenomena,
        families=families,
    )


def _accumulate_phenomena(
    details: dict[str, Any], tallies: dict[tuple[str, str], _Tally], gold: str
) -> None:
    """Fold one sample's marker/isogloss evidence into the running tallies."""
    seen: set[tuple[str, str]] = set()
    for group in _EVIDENCE_GROUPS:
        for item in details.get(group, ()):
            key = (str(item["phenomenon"]), str(item["label"]))
            tally = tallies.get(key)
            if tally is None:
                tally = tallies[key] = _Tally(tuple(item.get("families", ())))
            tally.total += int(item["count"])
            if key not in seen:
                seen.add(key)
                tally.carriers += 1
                tally.class_carriers[gold] = tally.class_carriers.get(gold, 0) + 1


def _accumulate_families(
    details: dict[str, Any], family_docs: dict[str, int], family_total: dict[str, int]
) -> None:
    """Fold one sample's positive per-family evidence into the running totals."""
    families = details.get("families", {})
    if not isinstance(families, dict):
        return
    for family, count in families.items():
        family_docs[family] = family_docs.get(family, 0) + 1
        family_total[family] = family_total.get(family, 0) + int(count)


def _phenomenon_frequency(
    key: tuple[str, str],
    tally: _Tally,
    class_size: dict[str, int],
    n_analysed: int,
    min_support: int,
) -> PhenomenonFrequency:
    """Score one tallied phenomenon into a :class:`PhenomenonFrequency`."""
    phenomenon, label = key
    scored = sorted(
        (
            (gold, count, (count * n_analysed) / (tally.carriers * class_size[gold]))
            for gold, count in tally.class_carriers.items()
        ),
        key=lambda item: (-item[2], -item[1], item[0]),
    )
    top_class, top_count, top_lift = scored[0]
    class_counts = tuple(
        ClassCount(label=gold, count=count)
        for gold, count in sorted(
            tally.class_carriers.items(), key=lambda item: (-item[1], item[0])
        )
    )
    return PhenomenonFrequency(
        phenomenon=phenomenon,
        label=label,
        families=tally.families,
        n_samples=tally.carriers,
        total_count=tally.total,
        top_class=top_class,
        top_class_lift=round(top_lift, EVIDENCE_FLOAT_DIGITS),
        top_class_share=round(top_count / tally.carriers, EVIDENCE_FLOAT_DIGITS),
        low_support=tally.carriers < min_support,
        class_counts=class_counts,
    )


def _rank_key(phenomenon: PhenomenonFrequency) -> tuple[bool, float, float, int, str, str]:
    """Total order for phenomena: reliable first, then by descending lift."""
    return (
        phenomenon.low_support,
        -phenomenon.top_class_lift,
        -phenomenon.top_class_share,
        -phenomenon.n_samples,
        phenomenon.phenomenon,
        phenomenon.label,
    )
