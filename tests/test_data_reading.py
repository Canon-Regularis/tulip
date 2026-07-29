"""Tests for tulip.data.reading.read_samples dispatch and errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.core.exceptions import DataError
from tulip.data.reading import read_samples

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no such file"):
        list(read_samples(tmp_path / "nope"))


def test_empty_jsonl_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert list(read_samples(path)) == []


def test_directory_is_read_as_a_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.csv").write_text(
        "id,text,speaker_id,dialect\ns1,baca hej kaj,spk,podhale\n", encoding="utf-8"
    )
    samples = list(read_samples(corpus))
    assert len(samples) == 1
    assert samples[0].text == "baca hej kaj"
