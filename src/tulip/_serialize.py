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
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from pydantic import BaseModel

__all__ = [
    "format_metric",
    "markdown_table",
    "round_floats",
    "save_report",
    "sorted_json_text",
    "tulip_version",
    "write_markdown",
    "write_sorted_json",
]


def format_metric(value: float | None, digits: int = 4) -> str:
    """Format a metric value for display, rendering ``None`` as ``"n/a"``.

    A pure formatting helper with no evaluation dependency, so it lives at the
    package root beside the shared writers rather than in ``evaluation``: the
    data and CLI layers render tables too and must not import ``evaluation``.

    Args:
        value: The metric value, or ``None`` when the metric is unavailable
            (e.g. ROC AUC without probability estimates).
        digits: Number of decimal places.

    Returns:
        The formatted string.
    """
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a GitHub-flavoured markdown table.

    The first column is left-aligned (names), remaining columns right-aligned
    (numbers), which keeps metric tables readable in rendered READMEs. Kept
    dependency-free so it never requires ``tabulate`` or any optional package.

    Args:
        headers: Column header cells.
        rows: Row cells; every row must have ``len(headers)`` entries.

    Returns:
        The markdown table as a single string (no trailing newline).
    """
    separators = [":---" if index == 0 else "---:" for index in range(len(headers))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separators) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def round_floats(payload: Any, digits: int) -> Any:
    """Recursively round every float in a JSON-native payload to ``digits`` places.

    Used before serialising a report so re-runs are byte-identical under trivial
    floating-point noise. Booleans pass through unchanged (a ``bool`` is not a
    ``float``, so this is explicit rather than load-bearing).
    """
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, float):
        return round(payload, digits)
    if isinstance(payload, dict):
        return {key: round_floats(value, digits) for key, value in payload.items()}
    if isinstance(payload, list):
        return [round_floats(item, digits) for item in payload]
    return payload


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


def save_report(model: BaseModel, path: Path | str, *, digits: int | None = None) -> None:
    """Write a pydantic report as deterministic JSON, optionally rounding floats.

    The one ``save()`` body every rigor report shares: dump the model to
    JSON-native values, round its floats to ``digits`` so re-runs are byte
    identical (skip when ``digits`` is ``None``), and write sorted-key JSON.
    Reports call this instead of repeating the three lines, which also keeps the
    byte-stability discipline in one place. A plain function, not a mixin, so it
    never touches a frozen model's MRO.

    Args:
        model: Any pydantic model exposing ``model_dump(mode="json")``.
        path: Destination file; parent directories are created.
        digits: Decimal places every float is rounded to for byte-stability, or
            ``None`` to write the dump unrounded.
    """
    payload = model.model_dump(mode="json")
    if digits is not None:
        payload = round_floats(payload, digits)
    write_sorted_json(Path(path), payload)


def tulip_version() -> str:
    """Return the installed tulip version, or ``"unknown"`` outside an install.

    The import is deferred so this module keeps importing nothing from ``tulip``
    at module load.
    """
    import tulip

    return getattr(tulip, "__version__", "unknown")
