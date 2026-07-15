"""Shared reading of the YAML resources that back the lexicon text features.

Both :func:`tulip.features.text.keywords.load_lexicon` and
:func:`tulip.features.text.phonology.load_isoglosses` begin identically: read a
bundled package-data file when no path is given (zip/Windows-safe via
:mod:`importlib.resources`), otherwise read the user's file and raise a clean
:class:`ConfigurationError` when it is absent, then ``yaml.safe_load`` the text.
:func:`read_yaml_resource` shares that prologue. The versioned-mapping loaders
(isoglosses and phonological rules) also share the version and entry-list checks
via :func:`read_versioned_entries`; only their per-entry validation stays local.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from typing import Any

__all__ = ["read_versioned_entries", "read_yaml_resource"]

#: Package and sub-directory the bundled feature lexicons ship under.
_RESOURCE_PACKAGE = "tulip.features.text"
_RESOURCE_DIR = "lexicons"


def read_yaml_resource(
    path: str | Path | None,
    *,
    bundled_name: str,
    noun: str,
    bundled_label: str | None = None,
) -> tuple[str, Any]:
    """Read and parse a bundled or user-supplied YAML resource.

    Args:
        path: A YAML file to read, or ``None`` to read the bundled resource.
        bundled_name: File name of the bundled resource under ``lexicons/``.
        noun: What the resource is called in a "not found" error (e.g.
            ``"lexicon"`` yields ``"lexicon file not found: ..."``).
        bundled_label: Word used in the bundled ``source`` label
            (``"bundled <label> '<name>'"``); defaults to ``noun``.

    Returns:
        ``(source, parsed)``: ``source`` is the human label each caller quotes
        in its own validation errors; ``parsed`` is the ``yaml.safe_load`` result
        (validated by the caller).

    Raises:
        ConfigurationError: if ``path`` is given but is not a file.
    """
    if path is None:
        label = bundled_label if bundled_label is not None else noun
        source = f"bundled {label} {bundled_name!r}"
        resource = resources.files(_RESOURCE_PACKAGE).joinpath(_RESOURCE_DIR).joinpath(bundled_name)
        raw = resource.read_text(encoding="utf-8")
    else:
        file_path = Path(path)
        source = str(file_path)
        if not file_path.is_file():
            raise ConfigurationError(f"{noun} file not found: {file_path}")
        raw = file_path.read_text(encoding="utf-8")
    return source, yaml.safe_load(raw)


def read_versioned_entries(
    path: str | Path | None,
    *,
    bundled_name: str,
    noun: str,
    bundled_label: str,
    entity: str,
    list_key: str,
    version: int,
) -> tuple[str, list[Any]]:
    """Read a versioned mapping resource and return its validated entry list.

    Wraps :func:`read_yaml_resource`, then checks the parsed content is a mapping
    carrying the expected ``version`` and a non-empty ``list_key`` list. Per-entry
    validation stays with each caller.

    Args:
        path: A YAML file to read, or ``None`` for the bundled resource.
        bundled_name: File name of the bundled resource under ``lexicons/``.
        noun: What the resource is called in a "not found" error.
        bundled_label: Word used in the bundled ``source`` label.
        entity: Singular name of one entry, used in the schema errors.
        list_key: Mapping key holding the entry list.
        version: The only schema version this loader accepts.

    Returns:
        ``(source, entries)``: the human label and the validated entry list.

    Raises:
        ConfigurationError: if the file is missing, is not a versioned mapping,
            reports the wrong version, or has an empty entry list.
    """
    source, data = read_yaml_resource(
        path, bundled_name=bundled_name, noun=noun, bundled_label=bundled_label
    )
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"{source}: {entity} file must be a mapping with 'version' and '{list_key}'"
        )
    if data.get("version") != version:
        raise ConfigurationError(
            f"{source}: unsupported {entity} schema version {data.get('version')!r}; "
            f"expected {version}"
        )
    entries = data.get(list_key)
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError(f"{source}: '{list_key}' must be a non-empty list")
    return source, entries
