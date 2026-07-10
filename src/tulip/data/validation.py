"""Structured integrity validation for manifest files.

Researchers assemble most tulip corpora by hand (see ``docs/datasets.md``),
and the synthetic generator writes manifests too. Before such a manifest is
fed to a training run, it is worth a cheap, actionable integrity check:
:func:`validate_manifest` parses it through the canonical
:func:`tulip.data.manifest.read_manifest` path and reports problems -- missing
content columns, out-of-taxonomy labels, absent audio files, and the
surrogate-speaker behaviour that silently shapes speaker-disjoint splitting --
as a :class:`ManifestReport` of typed :class:`ManifestIssue` records.

The report is deliberately CI-friendly: :attr:`ManifestReport.ok` is ``False``
only when a hard *error* is present, so warnings (out-of-enum labels,
surrogate over-grouping) do not fail a build, while genuinely unusable
manifests (no text/audio column, bad encoding) do. The severities are chosen
to match tulip's data contract: ``taxonomy.py`` explicitly permits
corpus-specific label strings to flow through, so those are *warnings*, never
errors.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from tulip.core.exceptions import DataError
from tulip.data.manifest import ManifestColumns, read_manifest
from tulip.labels.taxonomy import DialectFamily, RegionalDialect
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from tulip.core.types import Sample

_logger = get_logger(__name__)

Severity = Literal["error", "warning", "info"]

#: Ordering used to surface the most serious issues first in reports.
_SEVERITY_RANK: dict[str, int] = {"error": 0, "warning": 1, "info": 2}

#: Canonical taxonomy value sets, compared case-insensitively to mirror
#: :func:`tulip.labels.taxonomy.family_for` (which lower-cases before lookup).
_REGIONAL_DIALECT_VALUES = frozenset(member.value for member in RegionalDialect)
_DIALECT_FAMILY_VALUES = frozenset(member.value for member in DialectFamily)

#: Locality fields consulted for surrogate speaker IDs, in the exact order
#: :func:`tulip.data.manifest.read_manifest` hashes them.
_LOCALITY_FIELDS: tuple[str, ...] = ("village", "region", "voivodeship", "dialect")

#: Suffix-based format detection, mirrored from ``manifest.py`` so the shape
#: scan measures the file the same way the reader parses it. Kept local (rather
#: than importing manifest.py's private constants) to avoid coupling to another
#: module's internals; ``read_manifest`` remains the sole authority on parsing.
_CSV_DELIMITERS: dict[str, str] = {".csv": ",", ".tsv": "\t"}
_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})


class ManifestIssue(BaseModel):
    """A single integrity finding about a manifest.

    Attributes:
        severity: ``"error"`` fails validation; ``"warning"`` flags a probable
            problem that still permits loading; ``"info"`` is advisory.
        code: Stable machine-readable identifier (e.g. ``"unknown-dialect"``)
            suitable for filtering in CI.
        message: Human-readable explanation, including *why* it matters.
        row: 1-based source row when the finding is row-specific, else ``None``
            (many findings are file- or label-level rather than per row).
    """

    model_config = ConfigDict(frozen=True)

    severity: Severity
    code: str
    message: str
    row: int | None = None


class ManifestReport(BaseModel):
    """The outcome of validating one manifest file.

    Attributes:
        path: The validated manifest path, as a string for serialisation.
        n_rows: Total data rows found in the file (excluding the header).
        n_usable: Rows that parsed into a :class:`~tulip.core.types.Sample`
            (i.e. carried text and/or an audio path).
        counts: Aggregate tallies -- one entry per severity plus a
            ``"dialect:<label>"`` entry per observed dialect label.
        issues: All findings, most severe first.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    n_rows: int
    n_usable: int
    counts: dict[str, int]
    issues: tuple[ManifestIssue, ...]

    @property
    def ok(self) -> bool:
        """Whether the manifest is usable: ``True`` unless any issue is an error."""
        return not any(issue.severity == "error" for issue in self.issues)

    def to_markdown(self) -> str:
        """Render the report as GitHub-flavoured markdown.

        Mirrors :meth:`tulip.evaluation.report.EvaluationReport.to_markdown`:
        a summary table followed by an issues table built with the shared
        :func:`~tulip.evaluation._format.markdown_table` helper. Every issue's
        machine-readable ``code`` appears in the table so reports remain
        greppable.

        Returns:
            A markdown document ending without a trailing newline.
        """
        # Imported here, not at module scope: tulip.evaluation pulls in pandas,
        # and `import tulip.data` must not pay for a renderer most callers of
        # validate_manifest never invoke.
        from tulip.evaluation._format import markdown_table

        status = "OK" if self.ok else "FAILED"
        summary_rows = [
            ("Path", _escape_cell(self.path)),
            ("Status", status),
            ("Rows", str(self.n_rows)),
            ("Usable samples", str(self.n_usable)),
            ("Errors", str(self.counts.get("error", 0))),
            ("Warnings", str(self.counts.get("warning", 0))),
            ("Info", str(self.counts.get("info", 0))),
        ]
        parts = [
            "# Manifest validation report",
            markdown_table(("Field", "Value"), summary_rows),
        ]
        if self.issues:
            issue_rows = [
                (
                    issue.severity,
                    issue.code,
                    "" if issue.row is None else str(issue.row),
                    _escape_cell(issue.message),
                )
                for issue in self.issues
            ]
            parts.append("## Issues")
            parts.append(markdown_table(("Severity", "Code", "Row", "Message"), issue_rows))
        else:
            parts.append("No issues detected.")
        return "\n\n".join(parts)


def validate_manifest(
    path: Path,
    *,
    columns: ManifestColumns | None = None,
    audio_root: Path | None = None,
) -> ManifestReport:
    """Validate a manifest file and return a structured integrity report.

    The manifest is parsed through :func:`tulip.data.manifest.read_manifest`
    (the single canonical path), so structural failures -- missing required
    columns, "neither text nor audio", malformed rows, bad encoding -- surface
    as ``severity="error"`` issues rather than exceptions. On top of that, the
    following advisory checks run when parsing succeeds:

    * **Taxonomy** -- ``dialect``/``family`` values outside
      :class:`~tulip.labels.taxonomy.RegionalDialect`/:class:`DialectFamily`
      are *warnings*, not errors: corpus-specific labels are explicitly
      permitted to flow through the pipeline.
    * **Speaker IDs** -- reports whether the ``speaker_id`` column is present
      and, when it is absent, which fields surrogate IDs will be synthesised
      from, because over-grouping changes speaker-disjoint split behaviour.
    * **Audio** -- relative ``audio_path`` values are resolved against
      ``audio_root`` (default: the manifest's parent, matching
      ``read_manifest``); missing files are warnings.

    Args:
        path: The manifest file to validate.
        columns: Field-to-column mapping; defaults to
            :class:`ManifestColumns` defaults.
        audio_root: Base directory for relative audio paths; defaults to the
            manifest's parent directory.

    Returns:
        A :class:`ManifestReport`; call :attr:`ManifestReport.ok` for a
        pass/fail verdict.
    """
    path = Path(path)
    cols = columns if columns is not None else ManifestColumns()
    base = audio_root if audio_root is not None else path.parent
    issues: list[ManifestIssue] = []

    if not path.is_file():
        issues.append(
            ManifestIssue(
                severity="error",
                code="missing-file",
                message=f"manifest file not found: {path}",
            )
        )
        return _assemble(path, 0, 0, issues)

    # Authoritative parse: read_manifest owns all real parsing/validation, so
    # any failure it raises becomes the report's error(s).
    samples: list[Sample] | None
    try:
        samples = list(read_manifest(path, columns=cols, audio_root=base))
    except UnicodeDecodeError as exc:
        issues.append(
            ManifestIssue(
                severity="error",
                code="encoding",
                message=f"manifest is not valid UTF-8 (readers expect utf-8/utf-8-sig): {exc}",
            )
        )
        samples = None
    except DataError as exc:
        issues.append(
            ManifestIssue(
                severity="error",
                code=_classify_data_error(str(exc)),
                message=str(exc),
            )
        )
        samples = None

    # Best-effort shape scan (row count, columns, speaker coverage). Never a
    # source of parse errors: read_manifest already reported those above.
    n_rows, present_columns, n_with_speaker = _scan_shape(path, cols.speaker_id)

    if samples is None:
        return _assemble(path, n_rows, 0, issues)

    n_usable = len(samples)
    # read_manifest drops a row only when it has neither text nor audio, so once
    # a parse succeeds the shortfall is exactly those skipped rows.
    skipped = max(0, n_rows - n_usable)
    if skipped:
        issues.append(
            ManifestIssue(
                severity="warning",
                code="empty-rows",
                message=(
                    f"{skipped} of {n_rows} row(s) were skipped for having neither text "
                    "nor an audio path and produced no sample"
                ),
            )
        )

    issues.extend(_taxonomy_issues(samples))
    issues.append(_speaker_issue(cols, present_columns, n_rows, n_with_speaker))
    issues.extend(_audio_issues(samples))

    return _assemble(path, n_rows, n_usable, issues, samples)


def _assemble(
    path: Path,
    n_rows: int,
    n_usable: int,
    issues: list[ManifestIssue],
    samples: list[Sample] | None = None,
) -> ManifestReport:
    """Sort issues by severity, tally counts, and build the frozen report."""
    ordered = sorted(issues, key=lambda issue: _SEVERITY_RANK.get(issue.severity, 99))
    counts: dict[str, int] = dict.fromkeys(("error", "warning", "info"), 0)
    for issue in ordered:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
    for sample in samples or ():
        dialect = sample.labels.dialect
        if dialect:
            key = f"dialect:{dialect}"
            counts[key] = counts.get(key, 0) + 1
    return ManifestReport(
        path=str(path),
        n_rows=n_rows,
        n_usable=n_usable,
        counts=counts,
        issues=tuple(ordered),
    )


def _taxonomy_issues(samples: list[Sample]) -> list[ManifestIssue]:
    """Warn about dialect/family labels outside the recognised taxonomy.

    One warning per distinct out-of-enum value (with an occurrence count),
    since corpus-specific labels are permitted but will not map to a dialect
    family or to map geometry.
    """
    bad_dialects: dict[str, int] = {}
    bad_families: dict[str, int] = {}
    for sample in samples:
        dialect = sample.labels.dialect
        if dialect is not None and dialect.strip().lower() not in _REGIONAL_DIALECT_VALUES:
            bad_dialects[dialect] = bad_dialects.get(dialect, 0) + 1
        family = sample.labels.family
        if family is not None and family.strip().lower() not in _DIALECT_FAMILY_VALUES:
            bad_families[family] = bad_families.get(family, 0) + 1

    issues: list[ManifestIssue] = []
    for value, count in bad_dialects.items():
        issues.append(
            ManifestIssue(
                severity="warning",
                code="unknown-dialect",
                message=(
                    f"dialect {value!r} ({count} row(s)) is not a RegionalDialect value; "
                    "corpus-specific labels are allowed but will not resolve to a dialect "
                    "family or map region"
                ),
            )
        )
    for value, count in bad_families.items():
        issues.append(
            ManifestIssue(
                severity="warning",
                code="unknown-family",
                message=(
                    f"family {value!r} ({count} row(s)) is not a DialectFamily value; "
                    "corpus-specific labels are allowed but fall outside the taxonomy"
                ),
            )
        )
    return issues


def _speaker_issue(
    cols: ManifestColumns,
    present_columns: set[str],
    n_rows: int,
    n_with_speaker: int,
) -> ManifestIssue:
    """Describe speaker-ID coverage and any surrogate synthesis.

    Surrogate IDs shape speaker-disjoint splitting: replicating
    ``read_manifest``'s logic, when a row has no speaker it is grouped by its
    locality fields, or -- absent those -- collapsed to a single surrogate per
    file, which defeats leakage-free splitting entirely.
    """
    speaker_col = cols.speaker_id
    column_present = speaker_col is not None and speaker_col in present_columns
    locality_present = [
        field for field in _LOCALITY_FIELDS if getattr(cols, field) in present_columns
    ]

    if not column_present or n_with_speaker == 0:
        if locality_present:
            return ManifestIssue(
                severity="info",
                code="speaker-surrogate",
                message=(
                    "no speaker_id provided; stable surrogate speaker IDs will be "
                    f"synthesised by locality from {', '.join(locality_present)}. "
                    "Over-grouping only shrinks the number of split groups, so this is "
                    "safe but coarsens speaker-disjoint splitting"
                ),
            )
        return ManifestIssue(
            severity="warning",
            code="speaker-single-surrogate",
            message=(
                "no speaker_id and no locality fields "
                f"({', '.join(_LOCALITY_FIELDS)}); a single surrogate speaker will be "
                "synthesised per file, collapsing every row into one group and defeating "
                "speaker-disjoint splitting"
            ),
        )

    if n_with_speaker < n_rows:
        return ManifestIssue(
            severity="warning",
            code="partial-speaker",
            message=(
                f"{n_rows - n_with_speaker} of {n_rows} row(s) lack a speaker_id; "
                "surrogate IDs will be synthesised for those rows"
            ),
        )
    return ManifestIssue(
        severity="info",
        code="speaker-present",
        message=f"speaker_id is present for all {n_rows} row(s)",
    )


def _audio_issues(samples: list[Sample]) -> list[ManifestIssue]:
    """Warn about audio paths (already resolved by ``read_manifest``) that do not exist."""
    issues: list[ManifestIssue] = []
    for sample in samples:
        audio_path = sample.audio_path
        if audio_path is not None and not audio_path.exists():
            issues.append(
                ManifestIssue(
                    severity="warning",
                    code="missing-audio",
                    message=f"audio file not found for sample {sample.id!r}: {audio_path}",
                )
            )
    return issues


def _scan_shape(path: Path, speaker_col: str | None) -> tuple[int, set[str], int]:
    """Measure the manifest's shape without re-parsing it into samples.

    Counts data rows, collects the column names present, and tallies rows with
    a non-blank speaker cell -- the facts needed to report dropped rows and
    surrogate behaviour. All authoritative parsing stays in ``read_manifest``;
    this pass only mirrors its suffix-based format detection to *measure* the
    file, and swallows content errors (which ``read_manifest`` reports).

    Returns:
        ``(n_rows, present_columns, n_with_speaker)``; zeros/empties if the
        file cannot be read as expected.
    """
    suffix = path.suffix.lower()
    n_rows = 0
    n_with_speaker = 0
    present: set[str] = set()
    try:
        if suffix in _JSONL_SUFFIXES:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue  # read_manifest reports malformed JSON authoritatively
                    if not isinstance(record, dict):
                        continue
                    n_rows += 1
                    present.update(str(key) for key in record)
                    if speaker_col is not None and not _is_blank(record.get(speaker_col)):
                        n_with_speaker += 1
        else:
            delimiter = _CSV_DELIMITERS.get(suffix, ",")
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                if reader.fieldnames:
                    present.update(field for field in reader.fieldnames if field)
                for row in reader:
                    n_rows += 1
                    if speaker_col is not None and not _is_blank(row.get(speaker_col)):
                        n_with_speaker += 1
    except (OSError, UnicodeDecodeError) as exc:
        # read_manifest surfaces the authoritative error; the scan just degrades.
        _logger.debug("shape scan of %s degraded: %s", path, exc)
        return 0, set(), 0
    return n_rows, present, n_with_speaker


def _is_blank(value: Any) -> bool:
    """Whether a raw cell is empty (``None`` or whitespace-only)."""
    if value is None:
        return True
    text = value if isinstance(value, str) else str(value)
    return not text.strip()


def _classify_data_error(message: str) -> str:
    """Map a :class:`DataError` message to a stable issue code."""
    lowered = message.lower()
    if "neither a text column" in lowered or "neither text nor audio" in lowered:
        return "no-content-column"
    if "missing required column" in lowered:
        return "missing-column"
    if "no header row" in lowered or "is empty" in lowered:
        return "empty-manifest"
    if "invalid json" in lowered or "expected a json object" in lowered:
        return "malformed-row"
    if "invalid sample" in lowered:
        return "invalid-sample"
    return "parse-error"


def _escape_cell(text: str) -> str:
    """Escape pipe characters so a value cannot break markdown table rendering."""
    return text.replace("|", "\\|")


__all__ = ["ManifestIssue", "ManifestReport", "Severity", "validate_manifest"]
