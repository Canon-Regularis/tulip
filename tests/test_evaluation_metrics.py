"""Tests for tulip.evaluation.metrics.compute_metrics against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.metrics import compute_metrics

# Hand-computed micro-case:
#   y_true = a a b b, y_pred = a b b b
#   class a: tp=1 fp=0 fn=1 -> precision 1.0, recall 0.5, f1 2/3
#   class b: tp=2 fp=1 fn=0 -> precision 2/3, recall 1.0, f1 0.8
Y_TRUE = ["a", "a", "b", "b"]
Y_PRED = ["a", "b", "b", "b"]
# P(b) per sample: 0.1, 0.4, 0.35, 0.8 -> 3 of 4 (a, b) pairs ranked correctly -> AUC 0.75
Y_PROBA = [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]


def test_binary_micro_case_overall_metrics() -> None:
    report = compute_metrics(Y_TRUE, Y_PRED)
    assert report.accuracy == pytest.approx(0.75)
    assert report.balanced_accuracy == pytest.approx(0.75)
    assert report.precision_macro == pytest.approx(5 / 6)
    assert report.recall_macro == pytest.approx(0.75)
    assert report.f1_macro == pytest.approx((2 / 3 + 0.8) / 2)
    # Supports are equal (2 and 2), so weighted averages equal macro averages here.
    assert report.precision_weighted == pytest.approx(report.precision_macro)
    assert report.recall_weighted == pytest.approx(report.recall_macro)
    assert report.f1_weighted == pytest.approx(report.f1_macro)
    assert report.n_samples == 4


def test_binary_micro_case_per_class() -> None:
    report = compute_metrics(Y_TRUE, Y_PRED)
    a = report.per_class["a"]
    b = report.per_class["b"]
    assert a.precision == pytest.approx(1.0)
    assert a.recall == pytest.approx(0.5)
    assert a.f1 == pytest.approx(2 / 3)
    assert a.support == 2
    assert b.precision == pytest.approx(2 / 3)
    assert b.recall == pytest.approx(1.0)
    assert b.f1 == pytest.approx(0.8)
    assert b.support == 2


def test_labels_default_to_sorted_union_and_confusion_alignment() -> None:
    report = compute_metrics(["b", "b", "c"], ["b", "a", "c"])
    assert report.labels == ("a", "b", "c")
    # Rows = true, columns = predicted, in labels order.
    assert report.confusion == ((0, 0, 0), (1, 1, 0), (0, 0, 1))


def test_explicit_label_order_reorders_confusion() -> None:
    report = compute_metrics(Y_TRUE, Y_PRED, labels=["b", "a"])
    assert report.labels == ("b", "a")
    assert report.confusion == ((2, 0), (1, 1))


def test_zero_division_never_predicted_class_scores_zero() -> None:
    # "c" appears in labels and y_true but is never predicted: precision must be 0, not NaN.
    report = compute_metrics(["a", "b", "c"], ["a", "b", "b"], labels=["a", "b", "c"])
    assert report.per_class["c"].precision == 0.0
    assert report.per_class["c"].recall == 0.0
    assert report.per_class["c"].f1 == 0.0


def test_roc_auc_binary_hand_computed() -> None:
    report = compute_metrics(Y_TRUE, Y_PRED, y_proba=Y_PROBA)
    assert report.roc_auc_macro_ovr == pytest.approx(0.75)


def test_roc_auc_multiclass_perfect_separation() -> None:
    y_true = ["a", "b", "c"]
    proba = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    report = compute_metrics(y_true, y_true, y_proba=proba)
    assert report.roc_auc_macro_ovr == pytest.approx(1.0)


def test_roc_auc_none_without_proba() -> None:
    assert compute_metrics(Y_TRUE, Y_PRED).roc_auc_macro_ovr is None


def test_roc_auc_none_when_label_missing_from_y_true() -> None:
    proba = np.full((4, 3), 1 / 3)
    report = compute_metrics(Y_TRUE, Y_PRED, y_proba=proba, labels=["a", "b", "c"])
    assert report.roc_auc_macro_ovr is None


def test_roc_auc_none_on_column_count_mismatch() -> None:
    proba = np.full((4, 3), 1 / 3)  # 3 columns but only 2 labels
    report = compute_metrics(Y_TRUE, Y_PRED, y_proba=proba)
    assert report.roc_auc_macro_ovr is None


def test_roc_auc_none_on_non_numeric_or_ragged_proba() -> None:
    assert compute_metrics(Y_TRUE, Y_PRED, y_proba="nonsense").roc_auc_macro_ovr is None
    ragged = [[0.9, 0.1], [0.6], [0.65, 0.35], [0.2, 0.8]]
    assert compute_metrics(Y_TRUE, Y_PRED, y_proba=ragged).roc_auc_macro_ovr is None


def test_proba_row_count_mismatch_raises() -> None:
    with pytest.raises(ConfigurationError, match="rows"):
        compute_metrics(Y_TRUE, Y_PRED, y_proba=[[0.5, 0.5]] * 3)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ConfigurationError, match="same length"):
        compute_metrics(["a", "b"], ["a"])


def test_empty_input_raises() -> None:
    with pytest.raises(ConfigurationError, match="zero samples"):
        compute_metrics([], [])


def test_observed_label_missing_from_explicit_labels_raises() -> None:
    with pytest.raises(ConfigurationError, match="missing from"):
        compute_metrics(Y_TRUE, Y_PRED, labels=["a"])


def test_duplicate_labels_raise() -> None:
    with pytest.raises(ConfigurationError, match="duplicates"):
        compute_metrics(Y_TRUE, Y_PRED, labels=["a", "b", "a"])


def test_metadata_is_stored_and_numpy_inputs_accepted() -> None:
    report = compute_metrics(
        np.array(Y_TRUE),
        np.array(Y_PRED),
        metadata={"model": "naive_bayes", "split": "test", "target_level": "dialect"},
    )
    assert report.metadata["model"] == "naive_bayes"
    assert report.labels == ("a", "b")
    assert report.accuracy == pytest.approx(0.75)
