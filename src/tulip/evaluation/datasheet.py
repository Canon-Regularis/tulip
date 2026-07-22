"""A "Datasheets for Datasets" (Gebru et al.) document for a dialect corpus.

A benchmark is only credible if the data behind it is documented. This composes,
into one byte-stable markdown document, the facts a reviewer needs: the corpus's
provenance and licence, its split sizes and leakage-relevant speaker counts, its
class distribution at every taxonomy level, its geographic footprint, its
demographic composition, and the prose Gebru fields the static catalog cannot
carry (motivation, collection process, preprocessing, uses, distribution terms,
maintenance, ethical and legal considerations).

The prose lives in a :class:`DatasheetSpec` YAML sidecar the corpus author fills;
everything else is computed from the built splits with the same counting
discipline as :mod:`tulip.evaluation.cards`, so the document never claims a number
the data does not support. Missing fields degrade to ``"n/a"`` and never raise.

It lives in ``evaluation`` (not ``data``) because it composes
:mod:`tulip.evaluation.slicing` and the reporting helpers; the data layer must not
import ``evaluation``.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from tulip._serialize import markdown_table
from tulip.evaluation.slicing import record_slice_keys
from tulip.labels.geo import region_centroid, voivodeship_centroid
from tulip.labels.taxonomy import LabelLevel, display_name, family_for

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tulip.core.types import DatasetInfo, Sample
    from tulip.data.splitting import DatasetSplits

__all__ = ["DatasheetSpec", "datasheet", "load_datasheet_spec"]

_NA = "n/a"
_UNDOCUMENTED = "_Not documented._"
#: Every taxonomy level, tabulated in coarse-to-fine order.
_ALL_LEVELS: tuple[LabelLevel, ...] = tuple(LabelLevel)


class DatasheetSpec(BaseModel):
    """The prose Gebru fields the static catalog cannot carry (all optional)."""

    model_config = ConfigDict(frozen=True)

    motivation: str = ""
    funding: str = ""
    collection_process: str = ""
    sampling_strategy: str = ""
    preprocessing: str = ""
    uses: str = ""
    distribution_terms: str = ""
    maintenance_contact: str = ""
    ethical_legal: str = ""


def load_datasheet_spec(path: Path | str) -> DatasheetSpec:
    """Load a :class:`DatasheetSpec` from a YAML sidecar (an empty file is valid)."""
    from pathlib import Path

    from tulip.core.exceptions import ConfigurationError
    from tulip.utils.io import read_yaml

    try:
        raw = read_yaml(Path(path))
    except Exception as exc:  # missing / unparsable
        raise ConfigurationError(f"could not read datasheet spec {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"datasheet spec {path} must be a YAML mapping")
    from pydantic import ValidationError

    try:
        return DatasheetSpec.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid datasheet spec {path}: {exc}") from exc


def datasheet(
    info: DatasetInfo,
    splits: DatasetSplits,
    spec: DatasheetSpec,
    *,
    levels: Sequence[LabelLevel] = _ALL_LEVELS,
    conformance: str | None = None,
) -> str:
    """Render the datasheet as one byte-stable markdown document (no trailing newline).

    Args:
        info: Static corpus metadata (name, description, licence).
        splits: The built train/validation/test partitions.
        levels: Taxonomy levels to tabulate the class distribution at.
        spec: The prose sidecar.
        conformance: Optional pre-rendered ``tulip data validate`` markdown to
            embed as a data-quality section (the CLI supplies it when a source
            manifest is given).
    """
    samples = [sample for group in splits.as_dict().values() for sample in group]
    parts = [f"# Datasheet: {_text(info.name)}"]
    if str(info.description).strip():
        parts.append(str(info.description).strip())
    parts.append(_prose("Motivation", spec.motivation, spec.funding))
    parts.append(_composition_section(splits, samples, levels))
    parts.append(_geographic_section(samples))
    parts.append(_demographic_section(samples))
    parts.append(_prose("Collection process", spec.collection_process, spec.sampling_strategy))
    parts.append(_prose("Preprocessing", spec.preprocessing))
    parts.append(_prose("Uses", spec.uses))
    parts.append(_distribution_section(info, spec))
    parts.append(_prose("Maintenance", spec.maintenance_contact))
    parts.append(_prose("Ethical and legal considerations", spec.ethical_legal))
    if conformance and conformance.strip():
        parts.append(f"## Data quality and conformance\n\n{conformance.strip()}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------- sections


def _composition_section(
    splits: DatasetSplits, samples: list[Sample], levels: Sequence[LabelLevel]
) -> str:
    """Split sizes with distinct-speaker counts, then class distribution per level."""
    rows: list[tuple[str, ...]] = []
    for name, group in splits.as_dict().items():
        rows.append((name, str(len(group)), str(_distinct_speakers(group))))
    rows.append(("total", str(len(samples)), str(_distinct_speakers(samples))))
    lines = [
        "## Composition",
        f"{len(samples)} labelled instances, split speaker-disjoint so no speaker "
        "appears in more than one partition.",
        markdown_table(("Split", "Instances", "Distinct speakers"), rows),
    ]
    for level in levels:
        counts = Counter(
            label for sample in samples if (label := sample.labels.at_level(level)) is not None
        )
        if not counts:
            continue
        class_rows = [(display_name(label), str(counts[label])) for label in sorted(counts)]
        lines.append(
            f"### Class distribution: {level.value}\n\n"
            + markdown_table(("Class", "Instances"), class_rows)
        )
    return "\n\n".join(lines)


def _geographic_section(samples: list[Sample]) -> str:
    """Dialect-area and voivodeship footprints, joined to their WGS84 centroids.

    Only in-taxonomy labels (those with a known centroid) appear here; an
    out-of-taxonomy label string still counts in the class-distribution table but
    has no coordinate to place.
    """
    lines = [
        "## Geographic distribution",
        "Representative WGS84 centroids per area (dialect boundaries are gradients, not lines).",
    ]
    dialect_counts = Counter(
        label for sample in samples if (label := sample.labels.dialect) is not None
    )
    dialect_rows: list[tuple[str, ...]] = []
    for dialect in sorted(dialect_counts):
        centroid = region_centroid(dialect)
        if centroid is None:
            continue
        family = family_for(dialect)
        dialect_rows.append(
            (
                display_name(dialect),
                display_name(family.value) if family is not None else _NA,
                str(dialect_counts[dialect]),
                f"{centroid.lat:.2f}",
                f"{centroid.lon:.2f}",
            )
        )
    if dialect_rows:
        lines.append(
            "### Dialect areas\n\n"
            + markdown_table(("Dialect", "Family", "Instances", "Lat", "Lon"), dialect_rows)
        )

    voivodeship_counts = Counter(
        label for sample in samples if (label := sample.labels.voivodeship) is not None
    )
    voivodeship_rows: list[tuple[str, ...]] = []
    for voivodeship in sorted(voivodeship_counts):
        centroid = voivodeship_centroid(voivodeship)
        if centroid is None:
            continue
        voivodeship_rows.append(
            (
                voivodeship,
                str(voivodeship_counts[voivodeship]),
                f"{centroid.lat:.2f}",
                f"{centroid.lon:.2f}",
            )
        )
    if voivodeship_rows:
        lines.append(
            "### Voivodeships\n\n"
            + markdown_table(("Voivodeship", "Instances", "Lat", "Lon"), voivodeship_rows)
        )
    if not dialect_rows and not voivodeship_rows:
        lines.append("No in-taxonomy geographic labels are present.")
    return "\n\n".join(lines)


def _demographic_section(samples: list[Sample]) -> str:
    """Age-band and gender tallies read from each sample's metadata."""
    keys = [record_slice_keys(sample) for sample in samples]
    lines = ["## Demographic composition"]
    documented = False
    for dimension, heading in (("age_band", "Age band"), ("gender", "Gender")):
        counts = Counter(value for key in keys if (value := key[dimension]) is not None)
        if not counts:
            continue
        documented = True
        rows = [(value, str(counts[value])) for value in sorted(counts)]
        lines.append(f"### {heading}\n\n" + markdown_table((heading, "Instances"), rows))
    if not documented:
        lines.append("No demographic metadata (age, gender) is recorded for this corpus.")
    return "\n\n".join(lines)


def _distribution_section(info: DatasetInfo, spec: DatasheetSpec) -> str:
    """Licence (from the catalog) plus any distribution terms from the spec."""
    lines = ["## Distribution", f"Licence: {_text(getattr(info, 'license', None))}."]
    if spec.distribution_terms.strip():
        lines.append(spec.distribution_terms.strip())
    return "\n\n".join(lines)


# ---------------------------------------------------------------------- helpers


def _prose(heading: str, *paragraphs: str) -> str:
    """A prose section, or an explicit not-documented note when every field is blank."""
    body = "\n\n".join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
    return f"## {heading}\n\n{body or _UNDOCUMENTED}"


def _distinct_speakers(samples: Sequence[Sample]) -> int:
    """Count distinct non-empty speaker ids in ``samples``."""
    return len({sample.speaker_id for sample in samples if sample.speaker_id})


def _text(value: object) -> str:
    """Render a value for display, mapping ``None``/blank to ``"n/a"``."""
    if value is None:
        return _NA
    text = str(value).strip()
    return text or _NA
