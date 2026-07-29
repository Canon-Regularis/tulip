"""Tests for `tulip cite` and the version-parity guard."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tulip.cli._cite import (
    check_version_parity,
    find_repo_root,
    project_version,
    render_citation,
)
from tulip.cli.app import app
from tulip.core.exceptions import ConfigurationError

runner = CliRunner()

# tests/ sits directly under the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_metadata_versions_agree() -> None:
    assert check_version_parity(_REPO_ROOT) == []


def test_bibtex_carries_the_expected_fields() -> None:
    bibtex = render_citation("bibtex", _REPO_ROOT)
    assert bibtex.startswith("@software{tulip,")
    assert "Miezaniec, Matthew" in bibtex
    assert f"version = {{{project_version(_REPO_ROOT)}}}" in bibtex
    assert "year = {2026}" in bibtex


def test_apa_carries_the_expected_fields() -> None:
    apa = render_citation("apa", _REPO_ROOT)
    assert apa.startswith("Miezaniec, M. (2026).")
    assert "[Computer software]" in apa
    assert "github.com/Canon-Regularis/tulip" in apa


def test_unknown_style_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown citation style"):
        render_citation("chicago", _REPO_ROOT)


def test_parity_detects_drift(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "tulip"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\ntitle: tulip\nversion: 9.9.9\n"
        "date-released: 2020-01-01\nauthors:\n  - family-names: X\n    given-names: Y\n",
        encoding="utf-8",
    )
    drift = check_version_parity(tmp_path)
    assert len(drift) == 1
    assert "CITATION.cff" in drift[0]
    assert "9.9.9" in drift[0]


def test_find_repo_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == tmp_path.resolve()


def test_cite_command_prints_bibtex() -> None:
    result = runner.invoke(app, ["cite"])
    assert result.exit_code == 0, result.output
    assert "@software{tulip" in result.output


def test_cite_check_passes_on_committed_metadata() -> None:
    result = runner.invoke(app, ["cite", "--check"])
    assert result.exit_code == 0, result.output
    assert "agree" in result.output


# -------------------------------------------------------- root/version/parity


def test_find_repo_root_without_pyproject_raises(tmp_path) -> None:
    import pytest

    from tulip.cli._cite import find_repo_root

    with pytest.raises(ConfigurationError, match="pyproject"):
        find_repo_root(tmp_path)  # tmp_path and its parents hold no pyproject


def test_project_version_missing_version_raises(tmp_path) -> None:
    import pytest

    from tulip.cli._cite import project_version

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="version string"):
        project_version(tmp_path)


def test_load_citation_missing_or_non_mapping_raises(tmp_path) -> None:
    import pytest

    from tulip.cli._cite import load_citation

    with pytest.raises(ConfigurationError, match="no CITATION"):
        load_citation(tmp_path)
    (tmp_path / "CITATION.cff").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not a mapping"):
        load_citation(tmp_path)


def test_check_version_parity_flags_zenodo_drift(tmp_path) -> None:
    from tulip.cli._cite import check_version_parity

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        'cff-version: 1.2.0\nversion: "1.0.0"\n', encoding="utf-8"
    )
    (tmp_path / ".zenodo.json").write_text('{"version": "9.9.9"}', encoding="utf-8")
    drift = check_version_parity(tmp_path)
    assert any(".zenodo.json" in message for message in drift)
    assert not any("CITATION.cff" in message for message in drift)  # citation agrees
