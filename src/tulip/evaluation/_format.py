"""Shared formatting/serialisation helpers for evaluation artifacts.

Markdown rendering is kept dependency-free so it never requires ``tabulate`` or
any other optional package. :func:`write_sorted_json` is the one deterministic
JSON writer every committed evaluation artifact (leaderboard provenance,
prediction dumps, significance/selective reports) shares, so "byte-identical on
re-run" is guaranteed in exactly one place.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def format_metric(value: float | None, digits: int = 4) -> str:
    """Format a metric value for display, rendering ``None`` as ``"n/a"``.

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
    (numbers), which keeps metric tables readable in rendered READMEs.

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


def write_sorted_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as deterministic JSON (sorted keys, trailing newline).

    Sorted keys at every level and no timestamps mean re-serialising identical
    content is byte-identical, which is what makes a committed artifact
    regenerable and diffable. Mirrors the model-metadata sidecar contract.

    Args:
        path: Destination file; parent directories are created.
        payload: Any JSON-serialisable object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
