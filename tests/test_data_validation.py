"""Tests for the manifest integrity validator (``validate_manifest``).

Every manifest is written into ``tmp_path`` so the suite stays hermetic: no
network, no shared corpora. The severities are asserted precisely because
getting them wrong (e.g. erroring on a corpus-specific dialect label) is a
real contract bug -- ``taxonomy.py`` permits out-of-enum labels to flow
through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.data.manifest import ManifestColumns
from tulip.data.validation import ManifestReport, validate_manifest

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, rows: list[str]) -> Path:
    """Write newline-joined CSV/TSV ``rows`` as UTF-8 and return the path."""
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _warnings(report: ManifestReport) -> list[str]:
    return [issue.code for issue in report.issues if issue.severity == "warning"]


def _errors(report: ManifestReport) -> list[str]:
    return [issue.code for issue in report.issues if issue.severity == "error"]


def test_clean_manifest_is_ok_with_no_errors(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        [
            "id,text,speaker_id,dialect",
            "s1,Kaj żeś boł wczorej?,spk-a,silesia",
            "s2,Baca poseł na grań.,spk-b,podhale",
        ],
    )
    report = validate_manifest(manifest)
    assert report.ok is True
    assert _errors(report) == []
    assert report.n_rows == 2
    assert report.n_usable == 2


def test_out_of_enum_dialect_is_a_single_warning_not_an_error(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        [
            "text,speaker_id,dialect",
            "Wypowiedź pierwsza.,spk-a,martian",
            "Wypowiedź druga.,spk-b,martian",  # same bad value -> deduped to one warning
            "Wypowiedź trzecia.,spk-c,podhale",  # valid: no warning
        ],
    )
    report = validate_manifest(manifest)

    assert _errors(report) == []  # corpus-specific labels must never be errors
    assert report.ok is True
    assert _warnings(report) == ["unknown-dialect"]
    (issue,) = [i for i in report.issues if i.code == "unknown-dialect"]
    assert issue.severity == "warning"
    assert "martian" in issue.message


def test_out_of_enum_family_is_a_warning(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        [
            "text,speaker_id,family",
            "Jakiś tekst.,spk-a,klingon",
        ],
    )
    report = validate_manifest(manifest)
    assert _errors(report) == []
    assert "unknown-family" in _warnings(report)


def test_missing_audio_file_is_a_warning_but_present_audio_is_not(tmp_path: Path) -> None:
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "here.wav").write_bytes(b"RIFF")  # exists
    manifest = _write(
        tmp_path / "manifest.csv",
        [
            "id,audio_path,speaker_id,dialect",
            "a1,clips/here.wav,spk-a,podhale",
            "a2,clips/gone.wav,spk-b,podhale",  # missing on disk
        ],
    )
    report = validate_manifest(manifest)

    missing = [i for i in report.issues if i.code == "missing-audio"]
    assert len(missing) == 1
    assert missing[0].severity == "warning"
    assert "gone.wav" in missing[0].message


def test_audio_root_override_resolves_relative_paths(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    (audio_root / "clips").mkdir(parents=True)
    (audio_root / "clips" / "a.wav").write_bytes(b"RIFF")
    manifest = _write(
        tmp_path / "manifest.csv",
        ["id,audio_path,speaker_id", "a1,clips/a.wav,spk-a"],
    )
    report = validate_manifest(manifest, audio_root=audio_root)
    assert [i for i in report.issues if i.code == "missing-audio"] == []


def test_no_text_or_audio_column_fails_validation(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "manifest.csv", ["id,dialect", "s1,podhale"])
    report = validate_manifest(manifest)
    assert report.ok is False
    assert "no-content-column" in _errors(report)


def test_missing_required_column_is_an_error(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "manifest.csv", ["text,speaker_id", "abc,spk-1"])
    report = validate_manifest(manifest, columns=ManifestColumns(dialect="gwara"))
    assert report.ok is False
    assert "missing-column" in _errors(report)


def test_missing_speaker_column_reports_surrogate_info(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        [
            "text,dialect",
            "Wypowiedź z Podhala.,podhale",
            "Inna wypowiedź.,podhale",
        ],
    )
    report = validate_manifest(manifest)
    surrogate = [i for i in report.issues if i.code == "speaker-surrogate"]
    assert len(surrogate) == 1
    assert surrogate[0].severity == "info"
    assert "dialect" in surrogate[0].message  # names the field surrogates come from


def test_no_speaker_and_no_locality_is_a_warning(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        ["text", "Sama wypowiedź bez żadnych etykiet.", "Druga wypowiedź."],
    )
    report = validate_manifest(manifest)
    single = [i for i in report.issues if i.code == "speaker-single-surrogate"]
    assert len(single) == 1
    assert single[0].severity == "warning"


def test_rows_without_text_or_audio_are_counted_as_skipped(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        [
            "text,audio_path,speaker_id",
            "Prawdziwa wypowiedź.,,spk-a",
            ",,spk-b",  # neither text nor audio -> skipped by read_manifest
        ],
    )
    report = validate_manifest(manifest)
    assert report.n_rows == 2
    assert report.n_usable == 1
    empty = [i for i in report.issues if i.code == "empty-rows"]
    assert len(empty) == 1
    assert empty[0].severity == "warning"


def test_invalid_utf8_is_an_encoding_error(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"text,dialect\nba\xffz,podhale\n")  # \xff is not valid UTF-8
    report = validate_manifest(manifest)
    assert report.ok is False
    assert "encoding" in _errors(report)


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    report = validate_manifest(tmp_path / "nope.csv")
    assert report.ok is False
    assert "missing-file" in _errors(report)


def test_jsonl_manifest_is_supported(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"text": "Hej baca.", "speaker_id": "spk-1", "dialect": "podhale"}\n'
        '{"text": "Druga linia.", "speaker_id": "spk-2", "dialect": "martian"}\n',
        encoding="utf-8",
    )
    report = validate_manifest(manifest)
    assert report.n_rows == 2
    assert report.n_usable == 2
    assert "unknown-dialect" in _warnings(report)
    assert _errors(report) == []


def test_to_markdown_is_non_empty_and_contains_issue_codes(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        [
            "text,dialect",
            "Wypowiedź testowa.,martian",  # -> unknown-dialect + speaker-surrogate
        ],
    )
    report = validate_manifest(manifest)
    markdown = report.to_markdown()

    assert markdown.strip()
    assert "# Manifest validation report" in markdown
    assert report.issues  # sanity: there is something to render
    for issue in report.issues:
        assert issue.code in markdown


def test_counts_expose_severity_and_dialect_tallies(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        [
            "text,speaker_id,dialect",
            "Jedna.,spk-a,silesia",
            "Dwie.,spk-b,silesia",
            "Trzy.,spk-c,podhale",
        ],
    )
    report = validate_manifest(manifest)
    assert report.counts["error"] == 0
    assert report.counts["dialect:silesia"] == 2
    assert report.counts["dialect:podhale"] == 1
