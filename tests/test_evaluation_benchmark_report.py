"""Tests for the paper-style benchmark report assembler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.core.exceptions import DataError
from tulip.evaluation.benchmark_report import benchmark_report

if TYPE_CHECKING:
    from pathlib import Path

_LEADERBOARD = (
    "| Experiment | Model | F1 (macro) |\n| :--- | ---: | ---: |\n| real | majority | 0.10 |\n"
)
_SIGNIFICANCE = "# Significance: real\n\nmajority vs logistic_regression: p = 0.01\n"
_SECTIONS = (
    "## Abstract",
    "## Label hierarchy",
    "## Dataset",
    "## Protocol",
    "## Results",
    "## Significance",
    "## Demographic and geographic bias",
    "## Limitations",
)


def _board(tmp_path: Path) -> Path:
    board = tmp_path / "board"
    board.mkdir()
    (board / "leaderboard.md").write_text(_LEADERBOARD, encoding="utf-8")
    (board / "significance-real.md").write_text(_SIGNIFICANCE, encoding="utf-8")
    return board


def test_all_sections_present_and_byte_stable(tmp_path: Path) -> None:
    board = _board(tmp_path)
    doc = benchmark_report(board)
    for heading in _SECTIONS:
        assert heading in doc, heading
    assert doc.startswith("# Computational Identification of Polish Dialect Variation")
    assert "majority" in doc and "p = 0.01" in doc  # embedded board + significance
    assert "Lesser Polish" in doc and "Podhale" in doc  # taxonomy hierarchy table
    assert doc == benchmark_report(board)  # byte-stable


def test_synthetic_caption(tmp_path: Path) -> None:
    doc = benchmark_report(_board(tmp_path), synthetic=True)
    assert "Synthetic fixture, not real dialect accuracy" in doc


def test_embedded_datasheet_h1_is_demoted(tmp_path: Path) -> None:
    datasheet_md = "# Datasheet: dialektarium\n\n## Composition\n\n12 instances."
    doc = benchmark_report(_board(tmp_path), datasheet_md=datasheet_md)
    assert "### Datasheet: dialektarium" in doc  # demoted so the report keeps one H1
    assert "\n# Datasheet:" not in doc


def test_bias_section_embeds_or_points(tmp_path: Path) -> None:
    board = _board(tmp_path)
    with_bias = benchmark_report(board, bias_md="worst-vs-best gender gap: 0.20")
    assert "worst-vs-best gender gap: 0.20" in with_bias
    without = benchmark_report(board)
    assert "tulip analyze" in without and "--fairness" in without


def test_missing_leaderboard_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataError, match="no leaderboard"):
        benchmark_report(empty)
