"""Deterministic JSON serialisation shared across subsystems.

Several subsystems must write JSON that is *byte-identical* when the content is
identical: the leaderboard provenance, prediction dumps, significance and
selective reports, split locks, model sidecars, and the model registry index.
"Deterministic" means sorted keys at every level, a trailing newline, UTF-8, and
no timestamps, so re-serialising the same object reproduces the same bytes and a
committed or content-addressed artifact stays diff-friendly and hashable.

This lives at the package root, not in ``utils`` (frozen) or ``evaluation``
(which ``data`` and ``models`` must not import), so every layer can share the one
writer without a dependency cycle. It imports nothing from ``tulip``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

__all__ = ["sorted_json_text", "tulip_version", "write_markdown", "write_sorted_json"]


def sorted_json_text(payload: Any, *, default: Callable[[Any], Any] | None = None) -> str:
    """Serialise ``payload`` to deterministic JSON text (sorted keys, no newline).

    Args:
        payload: Any JSON-serialisable object.
        default: Optional fallback for values ``json`` cannot serialise natively
            (e.g. numpy scalars, ``Path``); ``None`` lets such values raise.

    Returns:
        The JSON string (two-space indented, sorted keys), without a trailing
        newline; callers that write to a file add one via
        :func:`write_sorted_json`.
    """
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=default)


def write_sorted_json(
    path: Path, payload: Any, *, default: Callable[[Any], Any] | None = None
) -> None:
    """Write ``payload`` to ``path`` as deterministic JSON with a trailing newline.

    Sorted keys and no timestamps make re-serialising identical content
    byte-identical, which is what keeps a committed or content-addressed artifact
    regenerable and diffable.

    Args:
        path: Destination file; parent directories are created.
        payload: Any JSON-serialisable object.
        default: Optional fallback for non-native values (see
            :func:`sorted_json_text`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        sorted_json_text(payload, default=default) + "\n", encoding="utf-8", newline="\n"
    )


def write_markdown(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8 markdown with one trailing newline.

    The trailing newline and fixed ``\\n`` line ending keep a committed report
    byte-identical when the content is identical, matching
    :func:`write_sorted_json`.

    Args:
        path: Destination file; parent directories are created.
        text: The markdown body, without a trailing newline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def tulip_version() -> str:
    """Return the installed tulip version, or ``"unknown"`` outside an install.

    The import is deferred so this module keeps importing nothing from ``tulip``
    at module load.
    """
    import tulip

    return getattr(tulip, "__version__", "unknown")
