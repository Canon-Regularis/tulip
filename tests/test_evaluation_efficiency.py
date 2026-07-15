"""Tests for tulip.evaluation.efficiency (machine-dependent cost metrics)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from conftest import make_samples
from tulip.evaluation.efficiency import (
    EfficiencyRecord,
    count_parameters,
    measure_efficiency,
    model_size_bytes,
    write_efficiency,
)
from tulip.pipeline import DialectClassifier

if TYPE_CHECKING:
    from pathlib import Path


def _fitted() -> DialectClassifier:
    return DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=0).fit(
        make_samples(repeats=4)
    )


# --------------------------------------------------------------- count_parameters


def test_count_parameters_reads_linear_coefficients() -> None:
    class _Linear:
        coef_ = np.zeros((3, 10))
        intercept_ = np.zeros(3)

    assert count_parameters(_Linear()) == 33  # 30 coefficients + 3 intercepts


def test_count_parameters_sums_over_a_pipeline() -> None:
    class _Countable:
        coef_ = np.zeros((1, 5))

    class _Opaque:
        pass

    class _Pipeline:
        def __init__(self) -> None:
            self.steps = [("vec", _Opaque()), ("clf", _Countable())]

    assert count_parameters(_Pipeline()) == 5


def test_count_parameters_is_none_when_undefined() -> None:
    class _Opaque:
        pass

    assert count_parameters(_Opaque()) is None


def test_count_parameters_counts_a_torch_like_module() -> None:
    class _Tensor:
        def __init__(self, n: int) -> None:
            self._n = n

        def numel(self) -> int:
            return self._n

    class _Module:
        def parameters(self) -> list[_Tensor]:
            return [_Tensor(4), _Tensor(6)]

    assert count_parameters(_Module()) == 10


# --------------------------------------------------------------- model_size_bytes


def test_model_size_bytes_sums_files(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 50)
    assert model_size_bytes(tmp_path) == 150


def test_model_size_bytes_is_none_for_a_missing_dir(tmp_path: Path) -> None:
    assert model_size_bytes(tmp_path / "nope") is None


# --------------------------------------------------------------- measure_efficiency


def test_measure_efficiency_populates_the_record() -> None:
    classifier = _fitted()
    samples = make_samples(repeats=1)
    record = measure_efficiency(classifier, samples, model="logistic_regression", repeats=2)
    assert record.model == "logistic_regression"
    assert record.n_samples == len(samples)
    assert record.latency_ms >= 0.0
    assert record.n_params is not None and record.n_params > 0  # linear coefficients
    assert record.model_size_bytes is None  # no model_dir given


def test_measure_efficiency_records_size_when_a_dir_is_given(tmp_path: Path) -> None:
    classifier = _fitted()
    model_dir = tmp_path / "model"
    classifier.save(model_dir)
    record = measure_efficiency(classifier, make_samples(repeats=1), model="m", model_dir=model_dir)
    assert record.model_size_bytes is not None and record.model_size_bytes > 0


def test_measure_efficiency_handles_an_empty_batch() -> None:
    record = measure_efficiency(_fitted(), [], model="m")
    assert record.n_samples == 0
    assert record.latency_ms == 0.0


# --------------------------------------------------------------- write_efficiency


def test_write_efficiency_sorts_and_round_trips(tmp_path: Path) -> None:
    records = [
        EfficiencyRecord(model="b", experiment="e2", n_samples=1, latency_ms=1.0),
        EfficiencyRecord(model="a", experiment="e1", n_samples=1, latency_ms=2.0),
    ]
    path = tmp_path / "efficiency.json"
    write_efficiency(records, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [r["experiment"] for r in payload] == ["e1", "e2"]  # sorted by (experiment, model)


def test_negative_latency_is_rejected() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        EfficiencyRecord(model="m", n_samples=1, latency_ms=-1.0)
