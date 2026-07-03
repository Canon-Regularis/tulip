"""Generic manifest reader turning CSV/TSV/JSONL rows into :class:`Sample` objects.

Most of the corpora tulip supports have no bulk download: the researcher
assembles them locally (per the acquisition notes in ``docs/datasets.md``)
into a *manifest* -- one row per sample, with columns for the text and/or an
audio file path plus whatever labels and speaker metadata the corpus
provides. This module is the single, well-tested path from such manifests to
validated :class:`~tulip.core.types.Sample` streams; corpus loaders simply
configure it with their column mapping and label defaults.

Column mapping is declarative via :class:`ManifestColumns`: each canonical
Sample field is mapped to a manifest column name, or to ``None`` to disable
it. Columns you configure explicitly must exist in the file (missing ones
raise :class:`~tulip.core.exceptions.DataError`); columns left at their
defaults are optional. Unmapped columns are preserved in ``Sample.metadata``
so no corpus information is silently dropped.

Speaker IDs are never left empty: when a row has no explicit speaker, a
stable surrogate is synthesised by hashing the available speaker-adjacent
metadata (village/region/voivodeship/dialect), falling back to one surrogate
per source file. Surrogates deliberately err toward *grouping* samples --
over-grouping merely shrinks the effective number of groups for splitting,
whereas under-grouping would leak speakers across splits.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from tulip.core.exceptions import DataError
from tulip.core.types import DialectLabels, Sample
from tulip.utils.logging import get_logger

_logger = get_logger(__name__)

#: Sample label fields a manifest column can populate.
_LABEL_FIELDS = ("family", "dialect", "region", "village", "voivodeship")

#: Delimiters inferred from manifest file suffixes.
_DELIMITERS = {".csv": ",", ".tsv": "\t"}

_JSONL_SUFFIXES = {".jsonl", ".ndjson"}


class ManifestColumns(BaseModel):
    """Mapping from canonical :class:`Sample` fields to manifest column names.

    Every field defaults to the like-named column; set a field to ``None`` to
    disable it (e.g. a corpus whose ``dialect`` column must be ignored).
    Fields you set explicitly are treated as required and raise
    :class:`DataError` when the column is absent from the manifest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str | None = "id"
    text: str | None = "text"
    audio_path: str | None = "audio_path"
    speaker_id: str | None = "speaker_id"
    family: str | None = "family"
    dialect: str | None = "dialect"
    region: str | None = "region"
    village: str | None = "village"
    voivodeship: str | None = "voivodeship"

    def required_columns(self) -> tuple[str, ...]:
        """Column names that were explicitly configured (and must exist)."""
        return tuple(
            sorted(
                str(getattr(self, field))
                for field in self.model_fields_set
                if getattr(self, field) is not None
            )
        )

    def mapped_columns(self) -> dict[str, str]:
        """Return ``{sample_field: column_name}`` for all enabled fields."""
        return {
            field: value
            for field in type(self).model_fields
            if (value := getattr(self, field)) is not None
        }


def surrogate_speaker_id(*parts: str) -> str:
    """Build a stable surrogate speaker ID from metadata fragments.

    The hash is deterministic across processes and platforms (unlike
    :func:`hash`), so splits computed today reproduce tomorrow.

    Args:
        parts: Any identifying fragments (source name, village, file name...).

    Returns:
        A short ``"spk-..."`` identifier stable for the given parts.
    """
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8)
    return f"spk-{digest.hexdigest()}"


def read_manifest(
    path: Path,
    *,
    columns: ManifestColumns | Mapping[str, str | None] | None = None,
    source: str = "manifest",
    label_defaults: Mapping[str, str] | None = None,
    audio_root: Path | None = None,
    delimiter: str | None = None,
) -> Iterator[Sample]:
    """Yield validated :class:`Sample` objects from a manifest file.

    Args:
        path: Manifest file; format inferred from the suffix (``.csv``,
            ``.tsv``, ``.jsonl``/``.ndjson``) unless ``delimiter`` overrides it.
        columns: Field-to-column mapping (:class:`ManifestColumns`, a plain
            mapping, or ``None`` for the defaults). Explicitly configured
            columns must exist in the file.
        source: Value for ``Sample.source`` and the surrogate-ID namespace.
        label_defaults: Label values applied when a row does not provide the
            field (e.g. ``{"dialect": "spisz"}`` for a single-dialect corpus).
        audio_root: Base directory for relative audio paths (defaults to the
            manifest's parent directory).
        delimiter: Explicit CSV delimiter, overriding suffix-based inference.

    Yields:
        One :class:`Sample` per usable row, in file order. Rows with neither
        text nor an audio path are skipped (logged at DEBUG).

    Raises:
        DataError: if the file is missing/malformed, an explicitly configured
            column is absent, or no text/audio column exists at all.
    """
    if not path.is_file():
        raise DataError(f"manifest file not found: {path}")
    cols = _coerce_columns(columns)
    defaults = dict(label_defaults or {})
    base = audio_root if audio_root is not None else path.parent

    jsonl = delimiter is None and path.suffix.lower() in _JSONL_SUFFIXES
    if jsonl:
        rows: Iterator[tuple[int, dict[str, Any]]] = _iter_jsonl_rows(path)
    else:
        rows = _iter_csv_rows(path, delimiter)
    required = cols.required_columns()

    checked_header = False
    for line_number, row in rows:
        if not checked_header:
            _check_columns(cols, row.keys(), path)
            checked_header = True
        elif jsonl and required:
            # CSV rows share one header, but JSONL records may be
            # heterogeneous: enforce explicitly-required columns per record,
            # not just on the first line.
            absent = [column for column in required if column not in row]
            if absent:
                raise DataError(
                    f"{path}:{line_number}: missing required column(s): {', '.join(absent)}"
                )
        sample = _row_to_sample(
            row,
            cols,
            source=source,
            defaults=defaults,
            audio_root=base,
            manifest_path=path,
            line_number=line_number,
        )
        if sample is None:
            _logger.debug("%s:%d: skipping row with neither text nor audio", path, line_number)
            continue
        yield sample


def _coerce_columns(
    columns: ManifestColumns | Mapping[str, str | None] | None,
) -> ManifestColumns:
    """Normalise the ``columns`` argument into a :class:`ManifestColumns`."""
    if columns is None:
        return ManifestColumns()
    if isinstance(columns, ManifestColumns):
        return columns
    try:
        return ManifestColumns(**dict(columns))
    except (TypeError, ValueError) as exc:
        raise DataError(f"invalid manifest column mapping {columns!r}: {exc}") from exc


def _iter_csv_rows(path: Path, delimiter: str | None) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, row_dict)`` from a delimited manifest.

    ``utf-8-sig`` tolerates the BOM that Excel prepends when saving CSV.
    """
    sep = delimiter or _DELIMITERS.get(path.suffix.lower(), ",")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=sep)
        if reader.fieldnames is None:
            raise DataError(f"manifest {path} is empty (no header row)")
        for row in reader:
            row.pop(None, None)  # extra unnamed cells beyond the header
            yield reader.line_num, {k: v for k, v in row.items() if k is not None}


def _iter_jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` from a JSON Lines manifest."""
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise DataError(f"{path}:{line_number}: expected a JSON object per line")
            yield line_number, record


def _check_columns(cols: ManifestColumns, available: Any, path: Path) -> None:
    """Validate the column mapping against the first row's keys."""
    present = set(available)
    missing = [c for c in cols.required_columns() if c not in present]
    if missing:
        raise DataError(
            f"manifest {path} is missing required column(s): {', '.join(missing)}; "
            f"available columns: {', '.join(sorted(present))}"
        )
    has_text = cols.text is not None and cols.text in present
    has_audio = cols.audio_path is not None and cols.audio_path in present
    if not has_text and not has_audio:
        raise DataError(
            f"manifest {path} has neither a text column ({cols.text!r}) nor an "
            f"audio path column ({cols.audio_path!r}); available columns: "
            f"{', '.join(sorted(present))}"
        )


def _cell(row: Mapping[str, Any], column: str | None) -> str | None:
    """Fetch a stripped cell value, mapping empty strings to ``None``."""
    if column is None:
        return None
    value = row.get(column)
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


def _row_to_sample(
    row: Mapping[str, Any],
    cols: ManifestColumns,
    *,
    source: str,
    defaults: Mapping[str, str],
    audio_root: Path,
    manifest_path: Path,
    line_number: int,
) -> Sample | None:
    """Convert one manifest row into a Sample (or ``None`` for empty rows)."""
    text = _cell(row, cols.text)
    audio_raw = _cell(row, cols.audio_path)
    if text is None and audio_raw is None:
        return None

    audio_path: Path | None = None
    if audio_raw is not None:
        candidate = Path(audio_raw)
        audio_path = candidate if candidate.is_absolute() else audio_root / candidate

    label_values: dict[str, str] = {}
    for field in _LABEL_FIELDS:
        value = _cell(row, getattr(cols, field))
        if value is None:
            value = defaults.get(field)
        if value is not None:
            label_values[field] = value

    speaker = _cell(row, cols.speaker_id)
    if speaker is None:
        location_parts = [
            label_values[field]
            for field in ("village", "region", "voivodeship", "dialect")
            if field in label_values
        ]
        if location_parts:
            speaker = surrogate_speaker_id(source, *location_parts)
        else:
            speaker = surrogate_speaker_id(source, manifest_path.name)

    sample_id = _cell(row, cols.id) or f"{source}-{manifest_path.stem}-{line_number:06d}"

    mapped = set(cols.mapped_columns().values())
    metadata = {
        key: value for key, value in row.items() if key not in mapped and value not in (None, "")
    }

    try:
        return Sample(
            id=sample_id,
            text=text,
            audio_path=audio_path,
            speaker_id=speaker,
            labels=DialectLabels(**label_values),
            source=source,
            metadata=metadata,
        )
    except ValueError as exc:
        raise DataError(f"{manifest_path}:{line_number}: invalid sample: {exc}") from exc


__all__ = ["ManifestColumns", "read_manifest", "surrogate_speaker_id"]
