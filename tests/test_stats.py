"""Tests for tulip._stats: the shared two-proportion test and Holm correction."""

from __future__ import annotations

import pytest

from tulip._stats import holm_correct, two_proportion_p


class TestTwoProportionP:
    def test_identical_rates_are_not_significant(self) -> None:
        # Same success rate in both groups: no evidence of a difference.
        assert two_proportion_p(5, 10, 50, 100) == pytest.approx(1.0)

    def test_a_wide_gap_on_ample_data_is_significant(self) -> None:
        assert two_proportion_p(90, 100, 10, 100) < 0.001

    def test_is_symmetric_in_its_two_groups(self) -> None:
        assert two_proportion_p(30, 100, 70, 100) == pytest.approx(
            two_proportion_p(70, 100, 30, 100)
        )

    def test_empty_group_returns_one(self) -> None:
        assert two_proportion_p(0, 0, 5, 10) == 1.0
        assert two_proportion_p(5, 10, 0, 0) == 1.0

    def test_no_variation_returns_one(self) -> None:
        # Every trial a success in both groups: nothing to separate them.
        assert two_proportion_p(10, 10, 20, 20) == 1.0
        assert two_proportion_p(0, 10, 0, 20) == 1.0


class TestHolmCorrect:
    def test_hand_computed_adjustment(self) -> None:
        # p = [0.01, 0.04, 0.03]; sorted 0.01, 0.03, 0.04 with factors 3, 2, 1
        # -> 0.03, 0.06, 0.06 (monotone), mapped back to input order.
        assert holm_correct([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_empty(self) -> None:
        assert holm_correct([]) == []

    def test_never_decreases_a_p_value(self) -> None:
        raw = [0.001, 0.2, 0.02, 0.5]
        adjusted = holm_correct(raw)
        assert all(a >= r for a, r in zip(adjusted, raw, strict=True))

    def test_is_clamped_to_one(self) -> None:
        assert all(value <= 1.0 for value in holm_correct([0.6, 0.7, 0.8]))
