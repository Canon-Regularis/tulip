"""Deterministic environment provenance for the reproducible leaderboard.

The leaderboard guarantee reads "for a fixed seed *and environment*", yet
``provenance.json`` records only ``tulip_version``: the environment half is
unstated. This module fills it with a block that is itself byte-identical across
machines, because every value is read from a *committed* file rather than the
live interpreter:

* the Python floor from ``pyproject.toml``;
* the locked versions of the numerically load-bearing dependencies (numpy,
  scipy, scikit-learn, pandas) from ``uv.lock``, the packages whose version can
  actually move a metric;
* content digests (BLAKE2b) of the exact configs and shipped lexicons that fed
  the run, so a silent edit to an input is caught.

Reading resolved versions from the lockfile, not ``importlib.metadata``, is the
point: two machines with the same commit produce the same block regardless of
what happens to be installed. If one of these inputs changes, the block changes
and the reproducibility gate flags it, which is correct, because a numpy bump
or an edited config can legitimately move the numbers.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomllib

from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["KEY_DEPENDENCIES", "environment_provenance"]

_logger = get_logger(__name__)

#: Dependencies whose version can move a leaderboard metric, so they are pinned
#: into provenance. Deliberately not the whole graph: tooling (ruff, pytest)
#: cannot change a result, and recording it would churn provenance for nothing.
KEY_DEPENDENCIES = ("numpy", "pandas", "scikit-learn", "scipy")

#: Shipped lexicons whose content shapes the linguistic features, hashed so an
#: edit is provable in the audit record.
_LEXICONS = ("dialect_markers.yaml", "isoglosses.yaml")

#: Compact digest size (bytes) for the BLAKE2b content hashes.
_DIGEST_BYTES = 16


def environment_provenance(
    config_paths: Sequence[Path], *, root: Path | None = None
) -> dict[str, Any]:
    """Build the deterministic ``environment`` block for provenance.

    Args:
        config_paths: The experiment configs that fed the run (hashed as inputs).
        root: Repository root holding ``pyproject.toml`` / ``uv.lock``; located
            by walking up from the working directory when omitted.

    Returns:
        A JSON-native mapping with sorted keys throughout: ``python`` floor,
        ``key_dependencies`` versions, and ``inputs`` content digests. Missing
        files degrade to ``null``/absent entries rather than raising, so
        provenance never fails an otherwise-valid run.
    """
    root = root or _find_root()
    return {
        "digest": f"blake2b-{_DIGEST_BYTES * 8}",
        "inputs": _input_digests(config_paths),
        "key_dependencies": _locked_versions(root / "uv.lock"),
        "python": _requires_python(root / "pyproject.toml"),
    }


def _find_root() -> Path:
    """Nearest ancestor of the CWD containing both project files (else the CWD)."""
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "uv.lock").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    return cwd


def _requires_python(pyproject: Path) -> str | None:
    """The ``requires-python`` floor from ``pyproject.toml`` (``None`` if absent)."""
    data = _read_toml(pyproject)
    if data is None:
        return None
    value = data.get("project", {}).get("requires-python")
    return str(value) if value is not None else None


def _locked_versions(lock_path: Path) -> dict[str, str]:
    """Locked versions of :data:`KEY_DEPENDENCIES` from ``uv.lock`` (sorted)."""
    data = _read_toml(lock_path)
    if data is None:
        return {}
    wanted = set(KEY_DEPENDENCIES)
    versions = {
        str(package["name"]): str(package["version"])
        for package in data.get("package", [])
        if isinstance(package, dict) and package.get("name") in wanted and "version" in package
    }
    return dict(sorted(versions.items()))


def _input_digests(config_paths: Sequence[Path]) -> dict[str, str]:
    """Content digests of the configs and shipped lexicons, keyed and sorted."""
    digests: dict[str, str] = {}
    for config_path in config_paths:
        path = Path(config_path)
        digest = _digest_bytes(_read_bytes(path))
        if digest is not None:
            digests[path.as_posix()] = digest
    for name in _LEXICONS:
        digest = _digest_lexicon(name)
        if digest is not None:
            digests[f"lexicon:{name}"] = digest
    return dict(sorted(digests.items()))


def _digest_lexicon(name: str) -> str | None:
    """Digest of a shipped lexicon resource, or ``None`` if it cannot be read."""
    try:
        data = (files("tulip.features.text.lexicons") / name).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        _logger.debug("lexicon %r not readable; omitted from provenance inputs", name)
        return None
    return _digest_bytes(data)


def _read_bytes(path: Path) -> bytes | None:
    """Read a file's bytes, or ``None`` (with a debug log) when it is absent."""
    try:
        return path.read_bytes()
    except OSError:
        _logger.debug("input %s not readable; omitted from provenance inputs", path)
        return None


def _digest_bytes(data: bytes | None) -> str | None:
    """BLAKE2b hex digest of ``data``, or ``None`` when ``data`` is ``None``."""
    if data is None:
        return None
    return hashlib.blake2b(data, digest_size=_DIGEST_BYTES).hexdigest()


def _read_toml(path: Path) -> dict[str, Any] | None:
    """Parse a TOML file, or ``None`` (with a debug log) when it is missing/broken."""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        _logger.debug("could not read %s for environment provenance", path)
        return None
