"""Tests for tulip.models._encoding: label encoding, fit-input validation,
balanced class weights, and the id<->label maps the fine-tuning heads consume."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tulip.core.exceptions import DataError
from tulip.models._encoding import (
    balanced_class_weights,
    encode_labels,
    label_id_maps,
    resolve_class_weights,
    validate_fit_inputs,
)


def test_encode_labels_sorts_classes_and_round_trips() -> None:
    classes, encoded = encode_labels(["b", "a", "b", "c"])
    assert list(classes) == ["a", "b", "c"]  # sorted unique
    assert encoded.dtype == np.int64
    # classes[encoded[i]] reconstructs str(y[i])
    assert [classes[i] for i in encoded] == ["b", "a", "b", "c"]


def test_encode_labels_coerces_to_str() -> None:
    classes, encoded = encode_labels([1, 2, 1])
    assert list(classes) == ["1", "2"]
    assert [classes[i] for i in encoded] == ["1", "2", "1"]


def test_validate_fit_inputs_accepts_a_well_formed_dataset() -> None:
    classes, encoded = validate_fit_inputs(["x", "y", "z"], ["a", "b", "a"])
    assert list(classes) == ["a", "b"]
    assert list(encoded) == [0, 1, 0]


def test_validate_fit_inputs_rejects_empty_inputs() -> None:
    with pytest.raises(DataError, match="empty"):
        validate_fit_inputs([], [])


def test_validate_fit_inputs_rejects_length_mismatch() -> None:
    with pytest.raises(DataError, match="length mismatch"):
        validate_fit_inputs(["x", "y"], ["a"])


def test_validate_fit_inputs_rejects_a_single_class() -> None:
    with pytest.raises(DataError, match="at least 2 classes"):
        validate_fit_inputs(["x", "y", "z"], ["a", "a", "a"])


def test_balanced_class_weights_follow_the_sklearn_formula() -> None:
    # counts [2, 1] over 3 samples, 2 classes: n / (n_classes * count).
    encoded = np.array([0, 0, 1])
    weights = balanced_class_weights(encoded, n_classes=2)
    np.testing.assert_allclose(weights, [3 / (2 * 2), 3 / (2 * 1)])


def test_balanced_class_weights_guard_a_class_absent_from_the_fold() -> None:
    # class 1 and 2 never appear; their count is floored to 1, not divided by 0.
    encoded = np.array([0, 0])
    weights = balanced_class_weights(encoded, n_classes=3)
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(weights, [2 / (3 * 2), 2 / (3 * 1), 2 / (3 * 1)])


def test_label_id_maps_are_mutually_inverse() -> None:
    classes, _ = encode_labels(["podhale", "silesia", "kurpie"])
    id2label, label2id = label_id_maps(classes)
    assert id2label == {0: "kurpie", 1: "podhale", 2: "silesia"}
    assert label2id == {"kurpie": 0, "podhale": 1, "silesia": 2}
    assert all(id2label[label2id[label]] == label for label in label2id)


def test_resolve_class_weights_honours_the_estimator_policy() -> None:
    encoded = np.array([0, 0, 1])
    balanced = resolve_class_weights(SimpleNamespace(class_weight="balanced"), encoded, 2)
    assert balanced is not None
    np.testing.assert_allclose(balanced, balanced_class_weights(encoded, 2))
    assert resolve_class_weights(SimpleNamespace(class_weight=None), encoded, 2) is None
    assert resolve_class_weights(SimpleNamespace(class_weight="none"), encoded, 2) is None
