"""Helpers shared by the text and audio synthetic corpus generators.

The two procedural generators (:mod:`tulip.data.synthetic` and
:mod:`tulip.data.synthetic_audio`) share three pieces of logic verbatim: the
validation of the knobs both specs carry, the resolution of requested dialect
keys against the available set, and the flattening of a :class:`Sample`'s labels
into a manifest record. This holds those three so the copies cannot drift, which
matters because both generators produce byte-stable corpora.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from tulip.core.types import Sample

__all__ = ["base_manifest_record", "resolve_dialect_keys", "validate_common_spec"]


def validate_common_spec(
    n_speakers_per_dialect: int, samples_per_speaker: int, dialects: tuple[str, ...] | None
) -> None:
    """Validate the knobs both synthetic specs carry, raising on any bad value.

    Each spec's ``__post_init__`` calls this, then checks its own extra knobs.

    Raises:
        ConfigurationError: if fewer than two speakers per dialect are requested
            (speaker-disjoint splitting needs it), fewer than one sample per
            speaker, or ``dialects`` is an empty sequence.
    """
    if n_speakers_per_dialect < 2:
        raise ConfigurationError(
            "n_speakers_per_dialect must be >= 2 so speaker-disjoint splitting "
            f"is meaningful, got {n_speakers_per_dialect}"
        )
    if samples_per_speaker < 1:
        raise ConfigurationError(f"samples_per_speaker must be >= 1, got {samples_per_speaker}")
    if dialects is not None and not dialects:
        raise ConfigurationError("dialects must be None or a non-empty sequence of keys")


def resolve_dialect_keys(
    dialects: tuple[str, ...] | None, available: tuple[str, ...], *, kind: str
) -> tuple[str, ...]:
    """Resolve requested dialect keys against ``available``, sorted and validated.

    ``None`` selects every available key. Otherwise each key is lowercased and
    checked; an unknown key raises :class:`ConfigurationError` naming ``kind`` and
    listing the available keys.

    Args:
        dialects: Requested keys, or ``None`` for all.
        available: The full set of valid keys, already sorted.
        kind: A short label for the corpus flavour, e.g. ``"synthetic"`` or
            ``"synthetic audio"``, used in the error message.

    Returns:
        The sorted, validated keys.

    Raises:
        ConfigurationError: if any requested key is not in ``available``.
    """
    if dialects is None:
        return available
    chosen = tuple(sorted({key.strip().lower() for key in dialects}))
    unknown = [key for key in chosen if key not in available]
    if unknown:
        raise ConfigurationError(
            f"unknown {kind} dialect key(s): {', '.join(unknown)}; "
            f"available keys: {', '.join(available)}"
        )
    return chosen


def base_manifest_record(sample: Sample) -> dict[str, object]:
    """The manifest fields both generators share: present labels, then the trailer.

    Returns the label fields that are set on ``sample`` (in taxonomy order)
    followed by the ``generator`` and ``spec_seed`` provenance trailer. Each
    generator prepends its own ``id``, modality field, and ``speaker_id`` via
    ``{"id": ..., "text"/"audio_path": ..., "speaker_id": ..., **base_manifest_record(sample)}``,
    so the key order stays identical and the corpus stays byte-stable.
    """
    record: dict[str, object] = {}
    for field in ("family", "dialect", "region", "village", "voivodeship"):
        value = getattr(sample.labels, field)
        if value is not None:
            record[field] = value
    record["generator"] = sample.metadata["generator"]
    record["spec_seed"] = sample.metadata["spec_seed"]
    return record
