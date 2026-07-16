"""Markdown and metric formatting helpers for evaluation artifacts (facade).

The helpers themselves are pure and layer-agnostic, so they live at the package
root in :mod:`tulip._serialize` alongside the deterministic JSON writer; the data
and CLI layers render tables too and must not import ``evaluation``. This module
re-exports them for the evaluation callers that already import them here, so the
move is transparent.
"""

from __future__ import annotations

from tulip._serialize import format_metric, markdown_table, write_sorted_json

__all__ = ["format_metric", "markdown_table", "write_sorted_json"]
