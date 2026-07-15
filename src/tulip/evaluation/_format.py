"""Shared markdown formatting helpers for evaluation artifacts.

Markdown rendering is kept dependency-free so it never requires ``tabulate`` or
any other optional package. The deterministic JSON writer that every committed
evaluation artifact shares (leaderboard provenance, prediction dumps,
significance/selective reports) lives at the package root in
:mod:`tulip._serialize`, so ``data`` and ``models`` can share the one writer
too. It is re-exported here for the evaluation callers that already import it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip._serialize import write_sorted_json

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["format_metric", "markdown_table", "write_sorted_json"]


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
