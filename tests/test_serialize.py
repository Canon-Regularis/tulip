"""Tests for tulip._serialize, the byte-stable serialisation bedrock.

Every committed leaderboard, provenance file, significance report, split lock,
and registry index is regenerated through these helpers, and the whole
reproducibility story rests on them writing byte-identical output for identical
content. That is worth pinning directly, not only through the higher-level
byte-stability tests: sorted keys at every level, floats rounded before
serialising, literal UTF-8 rather than escaped, and a fixed LF line ending on
every platform including this Windows one.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from tulip._serialize import (
    format_metric,
    markdown_table,
    round_floats,
    save_report,
    sorted_json_text,
    write_markdown,
    write_sorted_json,
)

if TYPE_CHECKING:
    from pathlib import Path

_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

# JSON-native payloads: exactly the value space model_dump(mode="json") yields,
# so the properties below exercise the same shapes the real reports produce.
_json = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-1000, max_value=1000)
    | st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
    | st.text(max_size=8),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(min_size=1, max_size=6), children, max_size=4)
    ),
    max_leaves=15,
)

# A z-with-dot and an a-with-ogonek: real Polish diacritics, built from code
# points so the source file stays ASCII and the assertion is unambiguous.
_POLISH = chr(0x17C) + chr(0x105)


# ------------------------------------------------------------------ format_metric


def test_format_metric_renders_none_as_na() -> None:
    assert format_metric(None) == "n/a"


def test_format_metric_uses_fixed_decimals_and_rounds() -> None:
    assert format_metric(0.5) == "0.5000"  # default is four places
    assert format_metric(0.5, digits=2) == "0.50"
    assert format_metric(0.123456, digits=3) == "0.123"
    assert format_metric(0.0) == "0.0000"
    assert format_metric(-1.25, digits=1) == "-1.2"  # banker's rounding


# ------------------------------------------------------------------ markdown_table


def test_markdown_table_left_aligns_the_first_column_and_right_aligns_the_rest() -> None:
    table = markdown_table(["Model", "F1", "Acc"], [["lr", "0.90", "0.88"], ["nb", "0.80", "0.79"]])
    assert table.splitlines() == [
        "| Model | F1 | Acc |",
        "| :--- | ---: | ---: |",
        "| lr | 0.90 | 0.88 |",
        "| nb | 0.80 | 0.79 |",
    ]
    assert not table.endswith("\n")  # callers add the newline via write_markdown


def test_markdown_table_with_no_rows_is_header_and_separator_only() -> None:
    table = markdown_table(["A", "B"], [])
    assert table.splitlines() == ["| A | B |", "| :--- | ---: |"]


def test_markdown_table_coerces_non_string_cells() -> None:
    table = markdown_table(["n", "p"], [[3, 0.5]])
    assert table.splitlines()[-1] == "| 3 | 0.5 |"


# ------------------------------------------------------------------ round_floats


def test_round_floats_rounds_nested_floats_only() -> None:
    payload = {"score": 0.123456, "counts": [1, 2, 3], "nested": {"f1": 0.98765}, "name": "lr"}
    assert round_floats(payload, 2) == {
        "score": 0.12,
        "counts": [1, 2, 3],
        "nested": {"f1": 0.99},
        "name": "lr",
    }


def test_round_floats_leaves_bools_ints_strings_and_none_untouched() -> None:
    # A bool is a subclass of int, so the explicit guard is what stops True from
    # being rounded to 1.0; ints, strings, and None are never floats to begin with.
    assert round_floats(True, 4) is True
    assert round_floats(False, 4) is False
    assert round_floats(7, 4) == 7
    assert round_floats("0.123456", 2) == "0.123456"
    assert round_floats(None, 2) is None


@_SETTINGS
@given(payload=_json, digits=st.integers(min_value=0, max_value=6))
def test_round_floats_is_idempotent(payload: object, digits: int) -> None:
    once = round_floats(payload, digits)
    assert round_floats(once, digits) == once


# ------------------------------------------------------------------ sorted_json_text


def test_sorted_json_text_sorts_keys_at_every_level() -> None:
    text = sorted_json_text({"b": 1, "a": {"z": 2, "y": 3}})
    assert text == '{\n  "a": {\n    "y": 3,\n    "z": 2\n  },\n  "b": 1\n}'
    assert not text.endswith("\n")  # no trailing newline; the file writer adds it


def test_sorted_json_text_keeps_unicode_literal_not_escaped() -> None:
    text = sorted_json_text({"dialect": _POLISH})
    assert _POLISH in text
    assert "\\u" not in text  # ensure_ascii=False, so no \uXXXX escapes


def test_sorted_json_text_raises_on_unserialisable_unless_default_given() -> None:
    class _Weird:
        pass

    with pytest.raises(TypeError):
        sorted_json_text({"w": _Weird()})
    text = sorted_json_text({"w": _Weird()}, default=lambda _obj: "weird")
    assert '"weird"' in text


@_SETTINGS
@given(data=st.dictionaries(st.text(min_size=1, max_size=6), st.integers(), max_size=8))
def test_sorted_json_text_is_key_order_independent(data: dict[str, int]) -> None:
    items = list(data.items())
    assert sorted_json_text(dict(items)) == sorted_json_text(dict(reversed(items)))


# ------------------------------------------------------------------ write_sorted_json


def test_write_sorted_json_ends_in_one_newline_and_round_trips(tmp_path: Path) -> None:
    payload = {"b": 2, "a": 1}
    path = tmp_path / "nested" / "out.json"
    write_sorted_json(path, payload)
    assert path.parent.is_dir()  # parents created
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert json.loads(text) == payload


def test_write_sorted_json_uses_lf_line_endings_on_every_platform(tmp_path: Path) -> None:
    # The reproducibility guarantee is byte-for-byte, so a CRLF slipping in on
    # Windows would break it. newline="\n" in the writer is what prevents that.
    path = tmp_path / "out.json"
    write_sorted_json(path, {"a": 1, "b": 2})
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert b"\r" not in raw


@_SETTINGS
@given(payload=_json)
def test_write_sorted_json_is_byte_deterministic(payload: object, tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    write_sorted_json(first, payload)
    write_sorted_json(second, payload)
    raw = first.read_bytes()
    assert raw == second.read_bytes()  # identical content, identical bytes
    assert b"\r\n" not in raw  # LF only, whatever the payload


# ------------------------------------------------------------------ write_markdown


def test_write_markdown_adds_one_newline_and_creates_parents(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "board.md"
    write_markdown(path, "line1\nline2")
    assert path.parent.is_dir()
    assert path.read_text(encoding="utf-8") == "line1\nline2\n"


def test_write_markdown_uses_lf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "board.md"
    write_markdown(path, "line1\nline2")
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"line2\n")


# ------------------------------------------------------------------ save_report


class _Report(BaseModel):
    name: str
    score: float
    per_class: dict[str, float]


def test_save_report_rounds_floats_when_digits_given(tmp_path: Path) -> None:
    report = _Report(name="lr", score=0.123456, per_class={"podhale": 0.98765})
    path = tmp_path / "report.json"
    save_report(report, path, digits=2)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {"name": "lr", "score": 0.12, "per_class": {"podhale": 0.99}}


def test_save_report_leaves_floats_unrounded_when_digits_is_none(tmp_path: Path) -> None:
    report = _Report(name="lr", score=0.123456, per_class={"podhale": 0.98765})
    path = tmp_path / "report.json"
    save_report(report, path, digits=None)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["score"] == 0.123456


def test_save_report_is_byte_stable_across_repeated_saves(tmp_path: Path) -> None:
    report = _Report(name="lr", score=0.123456, per_class={"b": 0.5, "a": 0.5})
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    save_report(report, first, digits=4)
    save_report(report, second, digits=4)
    assert first.read_bytes() == second.read_bytes()
