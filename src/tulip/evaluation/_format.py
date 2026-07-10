"""Shared plain-text formatting helpers for evaluation reports and tables.

Kept dependency-free so markdown rendering never requires ``tabulate`` or any
other optional package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


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
