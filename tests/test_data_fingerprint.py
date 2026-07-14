"""Tests for tulip.data.fingerprint: order-independent, drift-sensitive digests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import make_samples
from tulip.core.exceptions import DataError
from tulip.data.fingerprint import (
    SplitFingerprint,
    fingerprint_splits,
    split_digest,
    verify_splits,
)
from tulip.data.splitting import DatasetSplits

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def splits() -> DatasetSplits:
    samples = make_samples()
    return DatasetSplits(train=samples[:30], validation=samples[30:40], test=samples[40:])


class TestFingerprint:
    def test_is_deterministic(self, splits: DatasetSplits) -> None:
        assert fingerprint_splits(splits) == fingerprint_splits(splits)

    def test_split_digest_is_order_independent(self, splits: DatasetSplits) -> None:
        forward = split_digest(splits.train)
        reversed_ = split_digest(list(reversed(splits.train)))
        assert forward == reversed_

    def test_combined_digest_is_order_independent(self, splits: DatasetSplits) -> None:
        shuffled = DatasetSplits(
            train=list(reversed(splits.train)),
            validation=splits.validation,
            test=splits.test,
        )
        assert fingerprint_splits(shuffled).combined == fingerprint_splits(splits).combined

    def test_membership_change_changes_the_digest(self, splits: DatasetSplits) -> None:
        altered_first = splits.train[0].model_copy(update={"text": "zupełnie inny tekst"})
        altered = DatasetSplits(
            train=[altered_first, *splits.train[1:]],
            validation=splits.validation,
            test=splits.test,
        )
        assert fingerprint_splits(altered).combined != fingerprint_splits(splits).combined

    def test_sizes_are_recorded(self, splits: DatasetSplits) -> None:
        fingerprint = fingerprint_splits(splits)
        assert fingerprint.sizes == {"train": 30, "validation": 10, "test": 8}


class TestVerify:
    def test_passes_on_identical_splits(self, splits: DatasetSplits) -> None:
        verify_splits(splits, fingerprint_splits(splits))  # no raise

    def test_raises_naming_the_drifted_split(self, splits: DatasetSplits) -> None:
        expected = fingerprint_splits(splits)
        altered_test = splits.test[0].model_copy(update={"text": "przesunięcie testu"})
        drifted = DatasetSplits(
            train=splits.train,
            validation=splits.validation,
            test=[altered_test, *splits.test[1:]],
        )
        with pytest.raises(DataError, match="test:"):
            verify_splits(drifted, expected)


class TestPersistence:
    def test_save_load_round_trip(self, splits: DatasetSplits, tmp_path: Path) -> None:
        fingerprint = fingerprint_splits(splits)
        path = tmp_path / "split_lock.json"
        fingerprint.save(path)
        assert SplitFingerprint.load(path) == fingerprint

    def test_save_is_byte_identical(self, splits: DatasetSplits, tmp_path: Path) -> None:
        fingerprint = fingerprint_splits(splits)
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        fingerprint.save(first)
        fingerprint.save(second)
        assert first.read_bytes() == second.read_bytes()

    def test_load_rejects_non_lock_file(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.json"
        bogus.write_text('{"nope": 1}', encoding="utf-8")
        with pytest.raises(DataError, match="split lock"):
            SplitFingerprint.load(bogus)
