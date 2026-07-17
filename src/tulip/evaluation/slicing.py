"""Geographic and demographic slice keys for per-sample records.

The rigor analyses in :mod:`tulip.evaluation.error_analysis` and
:mod:`tulip.evaluation.fairness` slice per-sample outcomes to expose where a
model is weaker. They already slice by source, speaker, modality, and length.
This module adds the two axes a dialect benchmark's bias analysis needs:

* geographic: family, dialect, region, voivodeship, read straight from a
  sample's :class:`~tulip.core.types.DialectLabels` (the taxonomy the corpus was
  labelled against);
* demographic: age band and gender, read from ``Sample.metadata`` through a small
  alias table, because those fields are corpus-specific free-form columns (e.g.
  Common Voice carries ``age``/``gender``) rather than first-class label fields.

The keys are copied onto each :class:`~tulip.evaluation.predictions.PredictionRecord`
at scoring time, so downstream slicing never has to re-load the (possibly
unredistributable) corpus. Everything here is pure over
:class:`~tulip.core.types.Sample`, so it imports nothing from ``pipeline``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.labels.taxonomy import LabelLevel

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from tulip.core.types import Sample

__all__ = [
    "DEMOGRAPHIC_KEYS",
    "GEOGRAPHIC_LEVELS",
    "age_band",
    "bucket_by_upper_edge",
    "record_slice_keys",
]

#: The taxonomy levels surfaced as geographic slice dimensions (family is the
#: coarsest, voivodeship the geographic-administrative one). Village is omitted:
#: it is too fine to give a supported subgroup.
GEOGRAPHIC_LEVELS: tuple[LabelLevel, ...] = (
    LabelLevel.FAMILY,
    LabelLevel.DIALECT,
    LabelLevel.REGION,
    LabelLevel.VOIVODESHIP,
)

#: Demographic dimension -> accepted ``Sample.metadata`` column aliases. Corpora
#: spell these differently (Common Voice uses ``age``/``gender``), so a small
#: alias table keeps the slicing corpus-agnostic.
DEMOGRAPHIC_KEYS: dict[str, tuple[str, ...]] = {
    "gender": ("gender", "sex"),
    "age": ("age", "age_group", "age_band"),
}

#: Inclusive upper edges (in years) of the age bands, plus an open top. A numeric
#: age is bucketed; an already-banded string (e.g. Common Voice's "thirties")
#: passes through unchanged.
_AGE_BANDS: tuple[tuple[str, int | None], ...] = (
    ("<=19", 19),
    ("20-29", 29),
    ("30-44", 44),
    ("45-59", 59),
    ("60+", None),
)


def bucket_by_upper_edge(value: int, bands: tuple[tuple[str, int | None], ...]) -> str:
    """Return the label of the first band whose inclusive upper edge covers ``value``.

    ``bands`` runs coarse-to-fine as ``(label, upper)`` pairs, the last with an open
    top (``upper is None``) that catches everything larger. A value at or below a
    band's edge takes that band.
    """
    for label, upper in bands:
        if upper is None or value <= upper:
            return label
    return bands[-1][0]  # pragma: no cover - the open top always matches


def age_band(value: object) -> str:
    """Bucket a numeric age into a fixed band, or pass a non-numeric group through.

    A value that parses as a number is bucketed into :data:`_AGE_BANDS`. Anything
    else (an already-banded label such as ``"thirties"``, or a free-form group) is
    returned stripped and unchanged, so a corpus that ships pre-banded ages keeps
    its own bands.
    """
    text = str(value).strip()
    try:
        age = int(float(text))
    except (TypeError, ValueError):
        return text
    return bucket_by_upper_edge(age, _AGE_BANDS)


def _first_alias(metadata: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
    """Return the first non-empty aliased metadata value as a string, or ``None``."""
    for alias in aliases:
        value = metadata.get(alias)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def record_slice_keys(sample: Sample) -> dict[str, str | None]:
    """Extract the geographic and demographic slice keys for one sample.

    Returns a dict keyed exactly by the optional slice fields of
    :class:`~tulip.evaluation.predictions.PredictionRecord`
    (``region``/``voivodeship``/``family``/``dialect``/``age_band``/``gender``),
    so it can be splatted straight into the record constructor. A key is ``None``
    when the sample carries no value for it (an absent geographic level, or a
    corpus without that demographic column).
    """
    labels = sample.labels
    keys: dict[str, str | None] = {
        level.value: labels.at_level(level) for level in GEOGRAPHIC_LEVELS
    }
    gender = _first_alias(sample.metadata, DEMOGRAPHIC_KEYS["gender"])
    keys["gender"] = gender.lower() if gender is not None else None
    raw_age = _first_alias(sample.metadata, DEMOGRAPHIC_KEYS["age"])
    keys["age_band"] = age_band(raw_age) if raw_age is not None else None
    return keys
