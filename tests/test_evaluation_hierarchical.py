"""Tests for hierarchical family/dialect metrics."""

from __future__ import annotations

import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.hierarchical_metrics import PARTIAL_CREDIT_WEIGHT, hierarchical_metrics

# podhale and spisz are both Lesser Polish; kashubia is Kashubian.


def test_partial_credit_sits_between_exact_and_family() -> None:
    y_true = ["podhale", "podhale", "podhale", "podhale"]
    y_pred = ["podhale", "podhale", "spisz", "kashubia"]  # 2 exact, 1 family, 1 miss
    report = hierarchical_metrics(y_true, y_pred)
    assert report.exact_accuracy == pytest.approx(2 / 4)
    assert report.family_accuracy == pytest.approx(3 / 4)
    assert report.partial_credit == pytest.approx((2 * 1.0 + PARTIAL_CREDIT_WEIGHT) / 4)
    assert report.exact_accuracy <= report.partial_credit <= report.family_accuracy


def test_all_exact_is_perfect() -> None:
    report = hierarchical_metrics(["podhale", "silesia"], ["podhale", "silesia"])
    assert report.exact_accuracy == 1.0
    assert report.hierarchical_f1 == pytest.approx(1.0)


def test_family_only_hit_earns_partial_hierarchical_f1() -> None:
    report = hierarchical_metrics(["podhale"], ["spisz"])  # same family, wrong dialect
    assert report.exact_accuracy == 0.0
    assert report.family_accuracy == 1.0
    assert report.partial_credit == pytest.approx(PARTIAL_CREDIT_WEIGHT)
    assert report.hierarchical_f1 == pytest.approx(0.5)  # overlap 1 of sizes 2, 2


def test_family_mismatch_scores_zero() -> None:
    report = hierarchical_metrics(["podhale"], ["kashubia"])  # different families
    assert report.exact_accuracy == 0.0
    assert report.partial_credit == 0.0
    assert report.hierarchical_f1 == 0.0


def test_out_of_enum_labels_do_not_crash() -> None:
    report = hierarchical_metrics(["corpus_specific"], ["corpus_specific"])
    assert report.exact_accuracy == 1.0
    assert report.partial_credit == 1.0  # exact match still earns full credit


def test_rejects_bad_input() -> None:
    with pytest.raises(ConfigurationError, match="length"):
        hierarchical_metrics(["a"], ["a", "b"])
    with pytest.raises(ConfigurationError, match="at least one"):
        hierarchical_metrics([], [])


def test_markdown_names_the_metrics() -> None:
    markdown = hierarchical_metrics(["podhale"], ["podhale"]).to_markdown()
    assert "Hierarchical metrics" in markdown
    assert "hierarchical F1" in markdown
