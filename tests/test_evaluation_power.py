"""Tests for the minimum-detectable-effect power analysis."""

from __future__ import annotations

import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.power import minimum_detectable_effect


def test_mde_at_typical_n() -> None:
    # alpha=0.05 needs 6 one-directional wins: 2**-5 = 0.03125 <= 0.05 < 2**-4.
    report = minimum_detectable_effect(144, alpha=0.05, power=0.8)
    assert report.significant_wins == 6
    assert report.detectable
    assert report.mde is not None
    assert 0.04 < report.mde < 0.08


def test_small_n_is_undetectable() -> None:
    report = minimum_detectable_effect(3, alpha=0.05, power=0.8)
    assert not report.detectable
    assert report.mde is None
    assert report.significant_wins == 6  # needs 6 wins, only 3 samples


def test_significant_wins_track_alpha() -> None:
    # alpha=0.01 needs 8 wins: 2**-7 = 0.0078 <= 0.01 < 2**-6.
    assert minimum_detectable_effect(1000, alpha=0.01).significant_wins == 8


def test_mde_shrinks_with_more_samples() -> None:
    small = minimum_detectable_effect(100).mde
    large = minimum_detectable_effect(1000).mde
    assert small is not None
    assert large is not None
    assert large < small


def test_validation() -> None:
    with pytest.raises(ConfigurationError, match="n_samples"):
        minimum_detectable_effect(0)
    with pytest.raises(ConfigurationError, match="alpha"):
        minimum_detectable_effect(100, alpha=1.5)
    with pytest.raises(ConfigurationError, match="power"):
        minimum_detectable_effect(100, power=0.0)


def test_markdown_reports_the_result() -> None:
    assert "power" in minimum_detectable_effect(144).to_markdown().lower()
    assert "no accuracy gap" in minimum_detectable_effect(3).to_markdown().lower()


def test_binomial_tail_edge_cases() -> None:
    from tulip.evaluation.power import _binomial_tail

    assert _binomial_tail(10, 0, 0.5) == 1.0  # k <= 0: certain
    assert _binomial_tail(10, 11, 0.5) == 0.0  # k > n: impossible
    assert _binomial_tail(10, 3, 0.0) == 0.0  # p <= 0: no successes possible
    assert _binomial_tail(10, 3, 1.0) == 1.0  # p >= 1: every trial succeeds, k <= n
