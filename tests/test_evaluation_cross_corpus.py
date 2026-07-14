"""Tests for cross-corpus transfer evaluation (tulip.evaluation.cross_corpus)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import make_manifest_experiment_config, write_manifest_corpus
from tulip.core.exceptions import DataError
from tulip.core.types import DialectLabels, Sample
from tulip.evaluation import (
    CrossCorpusReport,
    TransferMatrix,
    partition_by_source,
    run_loco,
    transfer_matrix,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tulip.config import ExperimentConfig

_SENTENCES = {
    "podhale": ["baca poszedł na hale", "juhas widzioł owce na grani", "goról śpiywo po naszymu"],
    "silesia": ["jo żech je z katowic", "kaj żeś boł wczorej", "dej pozór na bajtla"],
    "kurpie": [
        "u nos psiwo warzą jesce",
        "kobziety śpsiewajo w kościele",
        "chłopoki poślo do lasu",
    ],
}


def _multi_source_samples(sources: tuple[str, ...] = ("corpusA", "corpusB")) -> list[Sample]:
    """A small multi-source, multi-speaker corpus over three dialects."""
    samples: list[Sample] = []
    for source in sources:
        for dialect, sentences in _SENTENCES.items():
            for speaker in range(3):
                for index, sentence in enumerate(sentences):
                    samples.append(
                        Sample(
                            id=f"{source}-{dialect}-{speaker}-{index}",
                            text=f"{sentence} wariant {speaker}",
                            speaker_id=f"{source}-{dialect}-{speaker}",
                            labels=DialectLabels(dialect=dialect),
                            source=source,
                        )
                    )
    return samples


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=3, variants=2)
    return make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="cc")


class TestPartition:
    def test_groups_labelled_samples_by_source(self) -> None:
        by_source = partition_by_source(_multi_source_samples(), "dialect")
        assert set(by_source) == {"corpusA", "corpusB"}
        assert all(s.source == "corpusA" for s in by_source["corpusA"])

    def test_drops_samples_without_a_target_label(self) -> None:
        samples = [Sample(id="x", text="tekst", source="c", labels=DialectLabels())]
        assert partition_by_source(samples, "dialect") == {}


class TestLoco:
    def test_reports_one_entry_per_held_out_corpus(self, config: ExperimentConfig) -> None:
        report = run_loco(config, samples=_multi_source_samples())
        assert isinstance(report, CrossCorpusReport)
        assert {r.held_out for r in report.results} == {"corpusA", "corpusB"}
        for result in report.results:
            assert result.n_train > 0 and result.n_test > 0
            assert 0.0 <= result.f1_macro <= 1.0
        assert 0.0 <= report.macro_f1 <= 1.0

    def test_requires_at_least_two_sources(self, config: ExperimentConfig) -> None:
        single = _multi_source_samples(sources=("only",))
        with pytest.raises(DataError, match="source corpora"):
            run_loco(config, samples=single)

    def test_markdown_renders(self, config: ExperimentConfig) -> None:
        report = run_loco(config, samples=_multi_source_samples())
        assert "Leave-one-corpus-out" in report.to_markdown()


class TestTransferMatrix:
    def test_full_grid(self, config: ExperimentConfig) -> None:
        matrix = transfer_matrix(config, samples=_multi_source_samples())
        assert isinstance(matrix, TransferMatrix)
        assert matrix.sources == ("corpusA", "corpusB")
        for train in matrix.sources:
            for test in matrix.sources:
                assert 0.0 <= matrix.score(train, test) <= 1.0
        assert 0.0 <= matrix.mean_off_diagonal <= 1.0

    def test_diagonal_beats_off_diagonal_on_average(self, config: ExperimentConfig) -> None:
        # In-sample (diagonal) should not be worse than transfer (off-diagonal).
        matrix = transfer_matrix(config, samples=_multi_source_samples())
        diagonal = sum(matrix.score(s, s) for s in matrix.sources) / len(matrix.sources)
        assert diagonal >= matrix.mean_off_diagonal

    def test_is_deterministic(self, config: ExperimentConfig) -> None:
        samples = _multi_source_samples()
        a = transfer_matrix(config, samples=samples)
        b = transfer_matrix(config, samples=samples)
        assert a.f1 == b.f1

    def test_markdown_renders(self, config: ExperimentConfig) -> None:
        matrix = transfer_matrix(config, samples=_multi_source_samples())
        assert "transfer matrix" in matrix.to_markdown()
