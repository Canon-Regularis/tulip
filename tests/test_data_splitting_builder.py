"""Tests for speaker-disjoint splitting and the dataset builder."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from conftest import DIALECT_TEMPLATES, make_samples, write_manifest_corpus
from tulip.config.schemas import ComponentConfig, DataConfig, SplitConfig
from tulip.core.exceptions import DataError
from tulip.core.types import DialectLabels, Sample
from tulip.data import DatasetBuilder, load_splits, speaker_disjoint_split
from tulip.data.builder import BUILD_MANIFEST_NAME
from tulip.labels.taxonomy import LabelLevel

if TYPE_CHECKING:
    from pathlib import Path


def _speakers(samples: list[Sample]) -> set[str]:
    return {s.speaker_id for s in samples if s.speaker_id}


class TestSpeakerDisjointSplit:
    def test_no_speaker_appears_in_two_splits(self) -> None:
        splits = speaker_disjoint_split(make_samples(repeats=4), SplitConfig(seed=42))
        train, val, test = (
            _speakers(splits.train),
            _speakers(splits.validation),
            _speakers(splits.test),
        )
        assert train and val and test
        assert not train & val
        assert not train & test
        assert not val & test
        assert splits.total == len(make_samples(repeats=4))

    def test_deterministic_for_a_seed_and_sensitive_to_it(self) -> None:
        samples = make_samples(repeats=6)
        first = speaker_disjoint_split(samples, SplitConfig(seed=7))
        second = speaker_disjoint_split(samples, SplitConfig(seed=7))
        assert [s.id for s in first.train] == [s.id for s in second.train]
        other = speaker_disjoint_split(samples, SplitConfig(seed=8))
        assert {s.id for s in other.train} != {s.id for s in first.train} or (
            _speakers(other.validation) != _speakers(first.validation)
        )

    def test_stratification_spreads_classes_across_splits(self) -> None:
        splits = speaker_disjoint_split(make_samples(repeats=8), SplitConfig(seed=42))
        train_dialects = {s.labels.dialect for s in splits.train if s.labels.dialect}
        # With 8 speakers per class, train (70%) must see every dialect.
        assert train_dialects == {"podhale", "silesia", "kurpie"}

    def test_empty_input_raises(self) -> None:
        with pytest.raises(DataError, match="empty"):
            speaker_disjoint_split([], SplitConfig())

    def test_save_load_accept_string_paths(self, tmp_path: Path) -> None:
        # Every other save/load pair in the toolkit accepts str; these must too.
        from tulip.data import load_splits, save_splits

        splits = speaker_disjoint_split(make_samples(repeats=4), SplitConfig(seed=42))
        save_splits(splits, str(tmp_path))
        assert load_splits(str(tmp_path)).sizes() == splits.sizes()

    def test_missing_speaker_id_raises(self) -> None:
        nameless = Sample(id="x", text="tekst bez mówcy", labels=DialectLabels())
        with pytest.raises(DataError, match="surrogate"):
            speaker_disjoint_split([nameless], SplitConfig())

    def test_too_few_groups_raises(self) -> None:
        one_speaker = [
            Sample(id=f"s{i}", text=f"zdanie {i}", speaker_id="only-one", labels=DialectLabels())
            for i in range(10)
        ]
        with pytest.raises(DataError, match="groups"):
            speaker_disjoint_split(one_speaker, SplitConfig())


@pytest.fixture
def manifest_corpus(tmp_path: Path) -> Path:
    """A manifest corpus (3 dialects x 4 speakers x 3 texts) plus two bad rows.

    The extras -- a too-short row and an exact duplicate -- must both be
    dropped by the builder's filter and dedup passes.
    """
    return write_manifest_corpus(
        tmp_path / "corpus",
        speakers=4,
        variants=3,
        extra_rows=(
            "short,za krótkie,podhale-spk0,podhale",
            f"dup,{DIALECT_TEMPLATES['podhale'].format(i='0-0')},podhale-spk0,podhale",
        ),
    )


class TestDatasetBuilder:
    def _config(self, root: Path) -> DataConfig:
        return DataConfig(
            datasets=[ComponentConfig(name="manifest", params={"root": str(root)})],
            root=root.parent,
            min_text_chars=20,
        )

    def test_build_end_to_end_with_persistence(self, manifest_corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "processed"
        # The near-duplicate pass is unit-tested in test_data_cleaning_dedup;
        # here it is disabled so the fixture's "wariant N" texts all survive
        # and the expected counts stay exact.
        builder = DatasetBuilder(
            self._config(manifest_corpus), dedup_params={"near_duplicates": False}
        )
        splits = builder.build(SplitConfig(seed=42), target=LabelLevel.DIALECT, output_dir=out)

        assert splits.total == 36  # 38 rows minus the short one and the duplicate
        assert not _speakers(splits.train) & _speakers(splits.test)
        assert not _speakers(splits.train) & _speakers(splits.validation)

        reloaded = load_splits(out)
        assert reloaded.sizes() == splits.sizes()
        assert [s.id for s in reloaded.train] == [s.id for s in splits.train]

        manifest = json.loads((out / BUILD_MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["sizes"] == splits.sizes()
        assert manifest["label_level"] == "dialect"
        assert set(manifest["class_distribution"]["train"]) <= {"podhale", "silesia", "kurpie"}
        assert manifest["data_config"]["min_text_chars"] == 20
        assert manifest["cleaner"]["lowercase"] is False

    def test_target_filter_raises_when_nothing_is_labelled(self, manifest_corpus: Path) -> None:
        builder = DatasetBuilder(self._config(manifest_corpus))
        with pytest.raises(DataError, match="village"):
            builder.build(SplitConfig(seed=1), target=LabelLevel.VILLAGE)

    def test_missing_dataset_root_raises_helpfully(self, tmp_path: Path) -> None:
        config = DataConfig(datasets=[ComponentConfig(name="dialektarium")], root=tmp_path)
        with pytest.raises(DataError, match=r"docs/datasets\.md"):
            DatasetBuilder(config).load_samples()

    def test_dedup_can_be_disabled(self, manifest_corpus: Path) -> None:
        config = self._config(manifest_corpus).model_copy(update={"deduplicate": False})
        samples = DatasetBuilder(config).load_samples()
        assert len(samples) == 37  # duplicate survives; short row still filtered
