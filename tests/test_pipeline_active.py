"""Tests for active-learning acquisition ranking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from conftest import make_samples
from tulip.core.exceptions import ConfigurationError
from tulip.core.registry import UnknownComponentError
from tulip.core.types import DialectLabels, Sample, TaskType
from tulip.pipeline import DialectClassifier
from tulip.pipeline.active import (
    STRATEGIES,
    AcquisitionContext,
    IntensityGated,
    rank_for_labeling,
)

if TYPE_CHECKING:
    from pathlib import Path

_LABELS = ("podhale", "silesia")


class _StubClassifier:
    """A fitted-classifier stand-in returning a fixed probability matrix."""

    task = TaskType.TEXT

    def __init__(self, proba: list[list[float]]) -> None:
        self.classes_ = np.array(_LABELS)
        self._proba = np.asarray(proba, dtype=np.float64)

    def predict_proba(self, raws: Any) -> np.ndarray:
        return self._proba[: len(list(raws))]


def _pool(*ids: str) -> list[Sample]:
    return [
        Sample(id=i, text=f"text for {i}", speaker_id="s", labels=DialectLabels(), source="pool")
        for i in ids
    ]


# --------------------------------------------------------------- strategy scores


def test_least_confidence_scores() -> None:
    context = AcquisitionContext(np.array([[0.9, 0.1], [0.5, 0.5]]), _LABELS, ["a", "b"])
    scores = STRATEGIES.create("least_confidence").score(context)
    assert scores == pytest.approx([0.1, 0.5])


def test_margin_scores() -> None:
    context = AcquisitionContext(np.array([[0.9, 0.1], [0.5, 0.5]]), _LABELS, ["a", "b"])
    scores = STRATEGIES.create("margin").score(context)
    assert scores == pytest.approx([1.0 - 0.8, 1.0 - 0.0])


def test_entropy_scores_are_normalised() -> None:
    context = AcquisitionContext(np.array([[1.0, 0.0], [0.5, 0.5]]), _LABELS, ["a", "b"])
    scores = STRATEGIES.create("entropy").score(context)
    assert scores == pytest.approx([0.0, 1.0])  # one-hot -> 0, uniform -> 1


def test_intensity_gated_zeroes_non_dialectal_text() -> None:
    # Both maximally uncertain; only the dialectal text should score above zero.
    proba = np.array([[0.5, 0.5], [0.5, 0.5]])
    raws = ["baca ma owce na hali juhas", "dzien dobry jak sie masz"]
    scores = IntensityGated().score(AcquisitionContext(proba, _LABELS, raws))
    assert scores[0] > 0.0  # dialectal markers present
    assert scores[1] == pytest.approx(0.0)  # bland standard Polish


def test_intensity_gated_rejects_non_text_inputs() -> None:
    from pathlib import Path

    proba = np.array([[0.5, 0.5]])
    context = AcquisitionContext(proba, _LABELS, [Path("clip.wav")])
    with pytest.raises(ConfigurationError, match="text inputs"):
        IntensityGated().score(context)


# --------------------------------------------------------------- ranking


def test_ranking_is_sorted_by_descending_score() -> None:
    # rows: confident, uncertain, middling -> uncertain ranks first.
    stub = _StubClassifier([[0.95, 0.05], [0.5, 0.5], [0.7, 0.3]])
    ranked = rank_for_labeling(stub, _pool("a", "b", "c"), strategy="least_confidence")  # type: ignore[arg-type]
    assert [c.sample_id for c in ranked] == ["b", "c", "a"]
    assert all(ranked[i].score >= ranked[i + 1].score for i in range(len(ranked) - 1))


def test_ties_break_by_sample_id() -> None:
    stub = _StubClassifier([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    ranked = rank_for_labeling(stub, _pool("z", "a", "m"), strategy="entropy")  # type: ignore[arg-type]
    assert [c.sample_id for c in ranked] == ["a", "m", "z"]  # equal score -> id order


def test_budget_keeps_the_top_n() -> None:
    stub = _StubClassifier([[0.95, 0.05], [0.5, 0.5], [0.7, 0.3]])
    ranked = rank_for_labeling(stub, _pool("a", "b", "c"), strategy="entropy", budget=2)  # type: ignore[arg-type]
    assert len(ranked) == 2
    assert ranked[0].sample_id == "b"


def test_budget_must_be_positive() -> None:
    stub = _StubClassifier([[0.5, 0.5]])
    with pytest.raises(ConfigurationError, match="budget"):
        rank_for_labeling(stub, _pool("a"), budget=0)  # type: ignore[arg-type]


def test_samples_without_the_modality_are_skipped() -> None:
    stub = _StubClassifier([[0.5, 0.5]])  # only one text sample survives
    text_sample = Sample(id="t", text="baca", speaker_id="s", labels=DialectLabels(), source="p")
    audio_only = Sample(
        id="a",
        audio_path="clip.wav",  # type: ignore[arg-type]
        speaker_id="s",
        labels=DialectLabels(),
        source="p",
    )
    ranked = rank_for_labeling(stub, [text_sample, audio_only], strategy="entropy")  # type: ignore[arg-type]
    assert [c.sample_id for c in ranked] == ["t"]


def test_unknown_strategy_raises() -> None:
    stub = _StubClassifier([[0.5, 0.5]])
    with pytest.raises(UnknownComponentError):
        rank_for_labeling(stub, _pool("a"), strategy="does_not_exist")  # type: ignore[arg-type]


def test_a_strategy_instance_is_accepted() -> None:
    stub = _StubClassifier([[0.5, 0.5]])
    ranked = rank_for_labeling(stub, _pool("a"), strategy=IntensityGated(min_intensity=0.5))  # type: ignore[arg-type]
    assert ranked[0].strategy == "intensity_gated"


def test_empty_pool_returns_empty() -> None:
    stub = _StubClassifier([[0.5, 0.5]])
    assert rank_for_labeling(stub, [], strategy="entropy") == []  # type: ignore[arg-type]


# --------------------------------------------------------------- integration


def test_ranking_over_a_real_classifier_is_deterministic() -> None:
    classifier = DialectClassifier(
        model="logistic_regression", features=["char_tfidf"], seed=0
    ).fit(make_samples(repeats=6))
    pool = _pool("u0", "u1", "u2", "u3")
    first = rank_for_labeling(classifier, pool, strategy="entropy")
    second = rank_for_labeling(classifier, pool, strategy="entropy")
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]
    assert {c.predicted_label for c in first} <= set(classifier.classes_)


# --------------------------------------------------------------- registry / hygiene


def test_all_strategies_are_registered() -> None:
    assert set(STRATEGIES.names()) == {
        "entropy",
        "intensity_gated",
        "least_confidence",
        "margin",
    }


def test_intensity_extractor_is_imported_lazily() -> None:
    # The extractor is imported inside intensity_gated's score(), not at module
    # level, so it is never a module global of active.py.
    import tulip.pipeline.active as active

    assert not hasattr(active, "DialectIntensityExtractor")


# --------------------------------------------------------------- CLI


def _saved_model_and_pool(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=0).fit(
        make_samples(repeats=3)
    ).save(model_dir)
    pool = _pool("u0", "u1", "u2")
    pool_path = tmp_path / "pool.jsonl"
    pool_path.write_text(
        "\n".join(sample.model_dump_json() for sample in pool) + "\n", encoding="utf-8"
    )
    return model_dir, pool_path


def test_acquire_command_runs(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tulip.cli.app import app

    model_dir, pool_path = _saved_model_and_pool(tmp_path)
    result = CliRunner().invoke(app, ["acquire", str(model_dir), str(pool_path), "--budget", "2"])
    assert result.exit_code == 0, result.output
    assert "acquisition ranking" in result.output


def test_acquire_json(tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from tulip.cli.app import app

    model_dir, pool_path = _saved_model_and_pool(tmp_path)
    result = CliRunner().invoke(
        app, ["acquire", str(model_dir), str(pool_path), "--strategy", "margin", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    assert {c["strategy"] for c in payload} == {"margin"}
    assert all("sample_id" in c and "score" in c for c in payload)


def test_acquire_unknown_strategy_lists_the_registered_options(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tulip.cli.app import app

    model_dir, pool_path = _saved_model_and_pool(tmp_path)
    result = CliRunner().invoke(
        app, ["acquire", str(model_dir), str(pool_path), "--strategy", "nope"]
    )
    assert result.exit_code == 1
    # The error enumerates the registry, not a hardcoded list.
    assert "entropy" in result.output
    assert "intensity_gated" in result.output
