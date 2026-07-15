"""``tulip cite``: citation strings and a version-parity guard.

A citable benchmark carries its citation metadata in the repository, and that
metadata must not drift from the package. This module reads the committed
``CITATION.cff`` and renders it as BibTeX or APA, and it checks that the version
stated in ``CITATION.cff`` and ``.zenodo.json`` still matches the package version
in ``pyproject.toml``.

``pyproject.toml`` is the single source of truth for the version. The check reads
all three from committed files (not the installed metadata), so it is hermetic
and gives the same answer in CI as on a laptop. A drift is an error, because a
release that bumps the package but forgets the citation files ships a wrong
citation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import tomllib
import yaml

from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "check_version_parity",
    "format_apa",
    "format_bibtex",
    "load_citation",
    "project_version",
    "render_citation",
]

_CITATION_FILE = "CITATION.cff"
_ZENODO_FILE = ".zenodo.json"


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` to the directory holding ``pyproject.toml``.

    Args:
        start: Directory to search from; the working directory when omitted.

    Returns:
        The repository root.

    Raises:
        ConfigurationError: if no ``pyproject.toml`` is found in any parent.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigurationError("could not locate pyproject.toml above the working directory")


def project_version(root: Path | None = None) -> str:
    """Read the canonical package version from ``pyproject.toml``.

    Raises:
        ConfigurationError: if the version is missing from ``[project]``.
    """
    root = root or find_repo_root()
    with (root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    version = data.get("project", {}).get("version")
    if not isinstance(version, str):
        raise ConfigurationError("pyproject.toml has no [project] version string")
    return version


def load_citation(root: Path | None = None) -> dict[str, Any]:
    """Parse ``CITATION.cff`` into a mapping.

    Raises:
        ConfigurationError: if the file is missing or is not a mapping.
    """
    root = root or find_repo_root()
    path = root / _CITATION_FILE
    if not path.is_file():
        raise ConfigurationError(f"no {_CITATION_FILE} at {root}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigurationError(f"{_CITATION_FILE} is not a mapping")
    return data


def check_version_parity(root: Path | None = None) -> list[str]:
    """Compare the citation files' versions against ``pyproject.toml``.

    Returns:
        A list of human-readable drift messages. Empty means every file agrees
        with the package version.
    """
    root = root or find_repo_root()
    canonical = project_version(root)
    drift: list[str] = []

    citation_version = load_citation(root).get("version")
    if str(citation_version) != canonical:
        drift.append(f"{_CITATION_FILE} version {citation_version!r} != pyproject {canonical!r}")

    zenodo_path = root / _ZENODO_FILE
    if zenodo_path.is_file():
        zenodo_version = json.loads(zenodo_path.read_text(encoding="utf-8")).get("version")
        if str(zenodo_version) != canonical:
            drift.append(f"{_ZENODO_FILE} version {zenodo_version!r} != pyproject {canonical!r}")

    return drift


def _authors(citation: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return ``(family, given)`` name pairs from the citation, in file order."""
    pairs: list[tuple[str, str]] = []
    for author in citation.get("authors", []):
        if isinstance(author, Mapping):
            pairs.append((str(author.get("family-names", "")), str(author.get("given-names", ""))))
    return pairs


def _year(citation: Mapping[str, Any]) -> str:
    """The release year, taken from ``date-released`` (``YYYY-MM-DD``)."""
    released = str(citation.get("date-released", ""))
    return released.split("-", 1)[0] if released else ""


def _initials(given: str) -> str:
    """Render given names as APA initials (``Anna Maria`` -> ``A. M.``)."""
    return " ".join(f"{part[0]}." for part in given.split() if part)


def _url(citation: Mapping[str, Any]) -> str:
    """The project URL, preferring ``url`` and falling back to ``repository-code``."""
    return str(citation.get("url") or citation.get("repository-code", ""))


def format_bibtex(citation: Mapping[str, Any]) -> str:
    """Render a ``@software`` BibTeX entry from parsed citation metadata."""
    authors = " and ".join(f"{family}, {given}".strip(", ") for family, given in _authors(citation))
    fields = {
        "author": authors,
        "title": str(citation.get("title", "")),
        "year": _year(citation),
        "version": str(citation.get("version", "")),
        "url": _url(citation),
        "license": str(citation.get("license", "")),
    }
    body = ",\n".join(f"  {key} = {{{value}}}" for key, value in fields.items() if value)
    return f"@software{{tulip,\n{body}\n}}"


def format_apa(citation: Mapping[str, Any]) -> str:
    """Render an APA reference string from parsed citation metadata."""
    names = "; ".join(
        f"{family}, {_initials(given)}".strip(", ") for family, given in _authors(citation)
    )
    year = _year(citation)
    title = str(citation.get("title", ""))
    version = str(citation.get("version", ""))
    url = _url(citation)
    version_note = f" (Version {version})" if version else ""
    return f"{names} ({year}). {title}{version_note} [Computer software]. {url}".strip()


def render_citation(style: str, root: Path | None = None) -> str:
    """Render the committed citation in ``bibtex`` or ``apa`` style.

    Raises:
        ConfigurationError: if ``style`` is not a supported format.
    """
    citation = load_citation(root)
    if style == "bibtex":
        return format_bibtex(citation)
    if style == "apa":
        return format_apa(citation)
    raise ConfigurationError(f"unknown citation style {style!r}; use 'bibtex' or 'apa'")
