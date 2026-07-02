"""Tests for :mod:`tulip.models.fasttext_model`.

Everything except the final smoke test runs without fastText installed: label
encoding, line formatting, and probability mapping are pure functions, and the
missing-dependency path is exercised by monkeypatching ``importlib``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from tulip.core.exceptions import (
    ConfigurationError,
    DataError,
    MissingDependencyError,
    TulipError,
)
from tulip.models import MODELS
from tulip.models.fasttext_model import (
    LABEL_PREFIX,
    FastTextClassifier,
    decode_fasttext_label,
    encode_fasttext_label,
    format_training_line,
    probability_row,
    sanitise_fasttext_text,
)


def block_imports(monkeypatch: pytest.MonkeyPatch, *blocked: str) -> None:
    """Make ``importlib.import_module`` fail for the given module trees."""
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if any(name == root or name.startswith(root + ".") for root in blocked):
            raise ImportError(f"blocked for test: {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


# --- registry / hyperparameter plumbing ----------------------------------------


def test_registry_contains_fasttext() -> None:
    assert "fasttext" in MODELS


def test_create_params_land_on_wrapper() -> None:
    model = MODELS.create(
        "fasttext",
        dim=50,
        epoch=10,
        lr=0.25,
        word_ngrams=3,
        min_count=2,
        minn=0,
        maxn=0,
        loss="ova",
        thread=2,
        seed=9,
    )
    assert isinstance(model, FastTextClassifier)
    assert model.dim == 50
    assert model.epoch == 10
    assert model.lr == pytest.approx(0.25)
    assert model.word_ngrams == 3
    assert model.min_count == 2
    assert model.minn == 0
    assert model.maxn == 0
    assert model.loss == "ova"
    assert model.thread == 2
    assert model.seed == 9


def test_defaults_are_deterministic_and_subword_aware() -> None:
    model = FastTextClassifier()
    assert model.thread == 1  # multithreaded fastText is non-deterministic
    assert model.dim == 100
    assert model.epoch == 25
    assert (model.minn, model.maxn) == (2, 5)
    assert model.seed == 42


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"epoch": 0},
        {"lr": 0.0},
        {"word_ngrams": 0},
        {"min_count": 0},
        {"minn": 5, "maxn": 2},
        {"minn": -1},
        {"thread": 0},
    ],
)
def test_constructor_rejects_bad_hyperparameters(kwargs: dict) -> None:
    with pytest.raises(ConfigurationError):
        FastTextClassifier(**kwargs)


# --- pure label / line helpers ---------------------------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "podhale",
        "with space",
        "łódź_żółć",
        "a/b?c%d",
        "__label__already",
        "mixed CASE-and-dash",
    ],
)
def test_label_encode_decode_round_trip(label: str) -> None:
    encoded = encode_fasttext_label(label)
    assert encoded.startswith(LABEL_PREFIX)
    assert " " not in encoded and "\t" not in encoded and "\n" not in encoded
    assert decode_fasttext_label(encoded) == label


def test_encode_label_coerces_non_strings() -> None:
    assert decode_fasttext_label(encode_fasttext_label(7)) == "7"


def test_decode_rejects_unprefixed_token() -> None:
    with pytest.raises(DataError):
        decode_fasttext_label("podhale")


def test_sanitise_collapses_all_whitespace() -> None:
    assert sanitise_fasttext_text("  hej\nbaco,\tkaj\r\nidziesz  ") == "hej baco, kaj idziesz"


def test_format_training_line_is_single_line() -> None:
    line = format_training_line("kaj\nidziesz", "with space")
    assert line == "__label__with%20space kaj idziesz"
    assert "\n" not in line


# --- probability mapping -----------------------------------------------------------


def test_probability_row_maps_and_orders() -> None:
    class_to_index = {"a": 0, "b": 1, "c": 2}
    row = probability_row(["__label__b", "__label__a"], [0.7, 0.3], class_to_index, 3)
    assert row == pytest.approx([0.3, 0.7, 0.0])
    assert row.sum() == pytest.approx(1.0)


def test_probability_row_clips_and_renormalises() -> None:
    class_to_index = {"a": 0, "b": 1}
    row = probability_row(["__label__a", "__label__b"], [1.0001, 0.5], class_to_index, 2)
    assert row.sum() == pytest.approx(1.0)
    assert row[0] == pytest.approx(1.0 / 1.5)


def test_probability_row_ignores_unknown_labels() -> None:
    row = probability_row(["__label__ghost", "__label__a"], [0.5, 0.5], {"a": 0}, 1)
    assert row == pytest.approx([1.0])


def test_probability_row_uniform_when_empty() -> None:
    row = probability_row([], [], {"a": 0, "b": 1, "c": 2, "d": 3}, 4)
    assert row == pytest.approx([0.25, 0.25, 0.25, 0.25])


# --- missing-dependency behaviour ---------------------------------------------------


def test_fit_without_fasttext_names_fasttext_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    block_imports(monkeypatch, "fasttext")
    model = MODELS.create("fasttext")
    with pytest.raises(MissingDependencyError) as excinfo:
        model.fit(["ala ma kota", "kaj ta idziesz"], ["standard", "silesia"])
    assert excinfo.value.extra == "fasttext"
    assert 'pip install "tulip[fasttext]"' in str(excinfo.value)


def test_predict_before_fit_raises_tulip_error() -> None:
    with pytest.raises(TulipError, match="not fitted"):
        FastTextClassifier().predict(["ala ma kota"])


# --- optional smoke test (skips cleanly when fasttext is absent) ----------------------


@pytest.mark.slow
def test_fasttext_train_predict_round_trip(synthetic_texts_and_labels) -> None:
    pytest.importorskip("fasttext")
    texts, labels = synthetic_texts_and_labels
    model = FastTextClassifier(dim=16, epoch=5, lr=0.5)
    model.fit(texts, labels)
    assert sorted(model.classes_.tolist()) == sorted(set(labels))
    probabilities = model.predict_proba(texts[:5])
    assert probabilities.shape == (5, len(model.classes_))
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(5), abs=1e-4)
    predictions = model.predict(texts[:5])
    assert all(prediction in set(labels) for prediction in predictions)
