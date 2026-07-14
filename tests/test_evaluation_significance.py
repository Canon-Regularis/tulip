"""Tests for tulip.evaluation.significance: exact McNemar, Holm, bootstrap CIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.predictions import PredictionRecord, SplitPredictions
from tulip.evaluation.significance import (
    _holm,
    mcnemar_exact,
    paired_significance,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestMcNemarExact:
    def test_hand_computed_two_sided_p(self) -> None:
        # a right/b wrong on 2 samples, never the reverse -> b=2, c=0.
        # exact two-sided p = 2 * C(2,0) * 0.5^2 = 0.5.
        discordant_a, discordant_b, p = mcnemar_exact(
            [True, True, True, False], [True, False, False, False]
        )
        assert (discordant_a, discordant_b) == (2, 0)
        assert p == pytest.approx(0.5)

    def test_strongly_discordant_is_significant(self) -> None:
        # 8 vs 0 discordant -> p = 2 * 0.5^8 = 0.0078125.
        _, _, p = mcnemar_exact([True] * 8 + [True] * 2, [False] * 8 + [True] * 2)
        assert p == pytest.approx(2 * 0.5**8)
        assert p < 0.05

    def test_no_discordant_pairs_is_p_one(self) -> None:
        _, _, p = mcnemar_exact([True, False], [True, False])
        assert p == 1.0

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ConfigurationError, match="same length"):
            mcnemar_exact([True], [True, False])


class TestHolm:
    def test_hand_computed_adjustment(self) -> None:
        # p = [0.01, 0.04, 0.03]; sorted 0.01,0.03,0.04 with factors 3,2,1
        # -> 0.03, 0.06, 0.06 (monotone), mapped back to input order.
        assert _holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_empty(self) -> None:
        assert _holm([]) == []


def _model(name: str, y_true: list[str], y_pred: list[str]) -> SplitPredictions:
    """A SplitPredictions with ids s0..sn over two classes."""
    records = tuple(
        PredictionRecord(
            id=f"s{i}",
            y_true=t,
            y_pred=p,
            proba=(0.9, 0.1) if p == "a" else (0.1, 0.9),
        )
        for i, (t, p) in enumerate(zip(y_true, y_pred, strict=True))
    )
    return SplitPredictions(model=name, split="test", labels=("a", "b"), records=records)


# 24 samples, alternating true labels.
Y_TRUE = ["a", "b"] * 12


class TestPairedSignificance:
    def test_best_model_and_tie_grouping(self) -> None:
        perfect = _model("perfect", Y_TRUE, Y_TRUE)  # every prediction correct
        one_off = list(Y_TRUE)
        one_off[0] = "b" if Y_TRUE[0] == "a" else "a"  # single error
        good = _model("good", Y_TRUE, one_off)
        bad = _model("bad", Y_TRUE, ["a"] * 24)  # predicts "a" always -> 50%

        report = paired_significance([perfect, good, bad], seed=0)

        assert report.best_model == "perfect"
        assert report.n_samples == 24
        tied = {m.model for m in report.models if m.tied_with_best}
        # 'good' differs from perfect on a single sample -> not significant -> tied.
        assert "good" in tied
        # 'bad' is wrong on ~half -> significantly worse -> not tied.
        assert "bad" not in tied

    def test_confidence_intervals_bracket_the_point(self) -> None:
        report = paired_significance(
            [_model("perfect", Y_TRUE, Y_TRUE), _model("bad", Y_TRUE, ["a"] * 24)],
            seed=0,
            n_resamples=200,
        )
        for model in report.models:
            for ci in model.metrics:
                assert ci.low <= ci.point <= ci.high

    def test_is_deterministic(self) -> None:
        models = [_model("perfect", Y_TRUE, Y_TRUE), _model("bad", Y_TRUE, ["a"] * 24)]
        a = paired_significance(models, seed=7, n_resamples=200)
        b = paired_significance(models, seed=7, n_resamples=200)
        assert a.model_dump() == b.model_dump()

    def test_save_is_byte_identical(self, tmp_path: Path) -> None:
        report = paired_significance(
            [_model("perfect", Y_TRUE, Y_TRUE), _model("bad", Y_TRUE, ["a"] * 24)],
            seed=0,
            n_resamples=200,
        )
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        report.save(first)
        report.save(second)
        assert first.read_bytes() == second.read_bytes()

    def test_rejects_unaligned_models(self) -> None:
        a = _model("a", Y_TRUE, Y_TRUE)
        shifted = SplitPredictions(
            model="b",
            split="test",
            labels=("a", "b"),
            records=tuple(
                PredictionRecord(id=f"x{i}", y_true=t, y_pred=t, proba=(0.9, 0.1))
                for i, t in enumerate(Y_TRUE)
            ),
        )
        with pytest.raises(ConfigurationError, match="aligned"):
            paired_significance([a, shifted])

    def test_requires_two_models(self) -> None:
        with pytest.raises(ConfigurationError, match="at least two"):
            paired_significance([_model("a", Y_TRUE, Y_TRUE)])
