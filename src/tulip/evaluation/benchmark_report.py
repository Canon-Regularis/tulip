"""Assemble a paper-style benchmark report from the harness's own outputs.

The review that motivated Phase 1 asked for a document, not just a leaderboard:
"Computational Identification of Polish Dialect Variation: A Speaker-Disjoint
Benchmark", covering the dataset methodology, the label hierarchy, the geographic
distribution, the demographic bias, the protocol, and the results.

This composes that document from artifacts already written elsewhere: the
datasheet (:mod:`tulip.evaluation.datasheet`), the committed ``leaderboard.md``
and ``significance-*.md`` from a board directory, an optional fairness/bias
section, and the static taxonomy and protocol prose. It recomputes nothing, so
the report is byte-stable and regenerates cleanly whenever its inputs change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip._serialize import markdown_table
from tulip.labels.taxonomy import (
    REGION_TO_FAMILY,
    DialectFamily,
    LabelLevel,
    display_name,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["DEFAULT_ABSTRACT", "DEFAULT_TITLE", "benchmark_report"]

DEFAULT_TITLE = (
    "Computational Identification of Polish Dialect Variation: A Speaker-Disjoint Benchmark"
)

DEFAULT_ABSTRACT = (
    "Polish dialect identification is a fine-grained classification problem "
    "grounded in phonology, morphology, lexicon, and geography. This report "
    "presents a reproducible, speaker-disjoint benchmark: models are trained and "
    "evaluated on frozen, content-fingerprinted splits in which no speaker appears "
    "in more than one partition, so a reported score reflects generalisation to "
    "unseen speakers rather than memorised voices. It documents the dataset, the "
    "label hierarchy, the geographic footprint, and the evaluation protocol, then "
    "reports each model's accuracy, macro-averaged F1, and calibration alongside "
    "paired significance tests against the majority-class floor."
)

#: The evaluation protocol, stated once so the report is byte-stable.
_PROTOCOL = (
    "Every model is trained and scored on one identical, frozen split. The split "
    "is speaker-disjoint and label-stratified: whole speaker groups are assigned "
    "to train, validation, or test, so no speaker crosses partitions and a score "
    "measures generalisation to unseen speakers, not recall of a memorised voice. "
    "The split is content-fingerprinted (a per-split BLAKE2b digest recorded in "
    "``split_lock.json``), so any silent change to the data is caught and the exact "
    "split behind a reported number can be reconstructed and verified.\n\n"
    "Reported metrics are accuracy, macro-averaged F1 (which weights every dialect "
    "equally, so a strong score cannot come from the majority class alone), "
    "weighted F1, one-versus-rest ROC AUC where probabilities allow it, and "
    "top-label calibration (expected calibration error and Brier score). Models are "
    "ranked by macro F1, ties broken deterministically, and each is compared to the "
    "majority-class floor with a paired bootstrap confidence interval and an exact "
    "McNemar test, Holm-corrected across the comparison set."
)

_LIMITATIONS = (
    "The taxonomy is a discrete approximation of a dialect continuum: real dialect "
    "boundaries are gradients, and a single hard label per sample understates that. "
    "Coverage is uneven across regions and speakers, so per-class and per-subgroup "
    "results with low support are flagged and must not be read as headline findings. "
    "Text-based identification cannot capture the phonetic cues that live only in "
    "audio, and self-reported or surrogate speaker and demographic metadata is "
    "imperfect. The benchmark measures identification accuracy under these "
    "constraints; it makes no claim about the sociolinguistic reality of any "
    "speaker or community."
)


def benchmark_report(
    board_dir: Path,
    *,
    title: str = DEFAULT_TITLE,
    abstract: str = DEFAULT_ABSTRACT,
    datasheet_md: str | None = None,
    bias_md: str | None = None,
    synthetic: bool = False,
) -> str:
    """Render the benchmark report as one byte-stable markdown document.

    Args:
        board_dir: A leaderboard output directory holding ``leaderboard.md`` and
            (optionally) ``significance-*.md``.
        title: Report title.
        abstract: Report abstract.
        datasheet_md: The corpus datasheet markdown to embed as the Dataset
            section; when ``None`` a pointer to ``tulip card datasheet`` is shown.
        bias_md: A fairness/bias analysis markdown (e.g. from
            ``tulip analyze --fairness``) to embed; when ``None`` a pointer is
            shown, because per-sample bias analysis needs predictions the board
            does not carry.
        synthetic: When ``True``, stamp a prominent caption that the results are a
            synthetic fixture, not real dialect accuracy.

    Raises:
        DataError: if ``board_dir`` has no ``leaderboard.md``.
    """
    from tulip.core.exceptions import DataError

    leaderboard_path = board_dir / "leaderboard.md"
    if not leaderboard_path.is_file():
        raise DataError(f"no leaderboard.md in board directory {board_dir}")
    leaderboard_md = leaderboard_path.read_text(encoding="utf-8").strip()
    significance_md = "\n\n".join(
        path.read_text(encoding="utf-8").strip()
        for path in sorted(board_dir.glob("significance-*.md"))
    )

    parts = [f"# {title}"]
    if synthetic:
        parts.append(
            "> **Synthetic fixture, not real dialect accuracy.** These numbers come "
            "from a procedurally generated corpus that exercises the pipeline; they "
            "say nothing about real dialect classification. See the caption on the "
            "results table."
        )
    parts.append(f"## Abstract\n\n{abstract}")
    parts.append(_label_hierarchy_section())
    parts.append(_dataset_section(datasheet_md))
    parts.append(f"## Protocol\n\n{_PROTOCOL}")
    parts.append(f"## Results\n\n{leaderboard_md}")
    if significance_md:
        parts.append(f"## Significance\n\n{significance_md}")
    parts.append(_bias_section(bias_md))
    parts.append(f"## Limitations\n\n{_LIMITATIONS}")
    return "\n\n".join(parts)


def _label_hierarchy_section() -> str:
    """The taxonomy: the five levels, then each family and its regional dialects."""
    levels = ", ".join(level.value for level in LabelLevel)
    by_family: dict[DialectFamily, list[str]] = {}
    for dialect, family in REGION_TO_FAMILY.items():
        by_family.setdefault(family, []).append(dialect.value)
    rows = []
    for family in sorted(by_family, key=lambda item: item.value):
        dialects = ", ".join(display_name(value) for value in sorted(by_family[family]))
        rows.append((display_name(family.value), dialects))
    families_without_regions = sorted(
        display_name(family.value) for family in DialectFamily if family not in by_family
    )
    lines = [
        "## Label hierarchy",
        f"Labels are hierarchical across {len(LabelLevel)} levels ({levels}); a "
        "family label auto-derives from a dialect label, and a corpus may carry "
        "labels at whichever levels it annotated. The regional dialects group into "
        "families as follows.",
        markdown_table(("Family", "Regional dialects"), rows),
    ]
    if families_without_regions:
        lines.append(
            "Families with no regional-dialect members in the taxonomy "
            f"(used at family level only): {', '.join(families_without_regions)}."
        )
    return "\n\n".join(lines)


def _dataset_section(datasheet_md: str | None) -> str:
    """Embed the datasheet, or point at how to generate one."""
    if datasheet_md and datasheet_md.strip():
        # Demote the datasheet's own H1 so the report keeps a single top level.
        body = datasheet_md.strip().replace("# Datasheet:", "### Datasheet:", 1)
        return f"## Dataset\n\n{body}"
    return (
        "## Dataset\n\nGenerate the corpus datasheet with "
        "`tulip card datasheet <build-dir> --spec <spec.yaml>` and embed it here; "
        "it documents provenance, splits and speaker counts, the class distribution "
        "at every level, the geographic footprint, and the demographic composition."
    )


def _bias_section(bias_md: str | None) -> str:
    """Embed the bias analysis, or point at how to produce it."""
    if bias_md and bias_md.strip():
        return f"## Demographic and geographic bias\n\n{bias_md.strip()}"
    return (
        "## Demographic and geographic bias\n\nSubgroup disparity is measured with "
        "`tulip analyze <predictions> --fairness`, which reports the best-versus-worst "
        "group gap over the geographic (region, voivodeship, family, dialect) and "
        "demographic (age band, gender) slices, with Holm-corrected two-proportion "
        "tests and low-support groups flagged. It runs on the per-sample predictions "
        "the board does not commit, so it is produced locally rather than embedded here."
    )
