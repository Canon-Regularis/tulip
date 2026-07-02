"""Tests for :mod:`tulip.models.neural_text` that need no optional dependency.

Heavy imports (torch, transformers) are exercised only through failure paths:
``importlib.import_module`` is monkeypatched so the heavy modules appear
uninstalled even on machines that have them.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError, MissingDependencyError, TulipError
from tulip.models import MODELS
from tulip.models.neural_text import (
    TransformerTextClassifier,
    balanced_class_weights,
    checkpoint_factory,
    encode_labels,
    linear_warmup_factor,
    optimizer_param_groups,
    require_fitted,
    resolve_device,
)


def block_imports(monkeypatch: pytest.MonkeyPatch, *blocked: str) -> None:
    """Make ``importlib.import_module`` fail for the given module trees."""
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if any(name == root or name.startswith(root + ".") for root in blocked):
            raise ImportError(f"blocked for test: {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


# --- factory / hyperparameter plumbing (no torch needed by contract) ---------


def test_factory_params_land_on_wrapper() -> None:
    model = MODELS.create(
        "herbert",
        max_length=128,
        epochs=5,
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_ratio=0.2,
        gradient_accumulation_steps=2,
        max_grad_norm=0.5,
        class_weight="balanced",
        device="cpu",
        seed=7,
    )
    assert isinstance(model, TransformerTextClassifier)
    assert model.checkpoint == "allegro/herbert-base-cased"
    assert model.max_length == 128
    assert model.epochs == 5
    assert model.batch_size == 4
    assert model.learning_rate == pytest.approx(1e-4)
    assert model.weight_decay == 0.0
    assert model.warmup_ratio == pytest.approx(0.2)
    assert model.gradient_accumulation_steps == 2
    assert model.max_grad_norm == pytest.approx(0.5)
    assert model.class_weight == "balanced"
    assert model.device == "cpu"
    assert model.seed == 7


def test_factory_checkpoint_is_overridable() -> None:
    model = MODELS.create("xlm_roberta", checkpoint="my-org/custom-roberta")
    assert model.checkpoint == "my-org/custom-roberta"


def test_defaults_follow_training_config_conventions() -> None:
    model = TransformerTextClassifier()
    assert model.max_length == 256
    assert model.epochs == 3
    assert model.batch_size == 16
    assert model.learning_rate == pytest.approx(2e-5)
    assert model.seed == 42


def test_checkpoint_factory_binds_and_forwards() -> None:
    factory = checkpoint_factory(TransformerTextClassifier, "some/checkpoint")
    model = factory(epochs=9)
    assert model.checkpoint == "some/checkpoint"
    assert model.epochs == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_length": 0},
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"warmup_ratio": 1.5},
        {"gradient_accumulation_steps": 0},
        {"class_weight": "boosted"},
    ],
)
def test_constructor_rejects_bad_hyperparameters(kwargs: dict) -> None:
    with pytest.raises(ConfigurationError):
        TransformerTextClassifier(**kwargs)


# --- missing-dependency behaviour ---------------------------------------------


def test_fit_without_torch_names_transformers_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    block_imports(monkeypatch, "torch", "transformers")
    model = MODELS.create("herbert")
    with pytest.raises(MissingDependencyError) as excinfo:
        model.fit(["ala ma kota", "kaj ta idziesz"], ["standard", "silesia"])
    assert excinfo.value.extra == "transformers"
    assert 'pip install "tulip[transformers]"' in str(excinfo.value)


def test_predict_before_fit_raises_tulip_error() -> None:
    model = TransformerTextClassifier()
    with pytest.raises(TulipError, match="not fitted"):
        model.predict(["ala ma kota"])


# --- pure helpers ---------------------------------------------------------------


def test_encode_labels_round_trip() -> None:
    classes, encoded = encode_labels(["b", "a", "b", "c", "a"])
    assert classes.tolist() == ["a", "b", "c"]
    assert encoded.dtype == np.int64
    assert [classes[i] for i in encoded] == ["b", "a", "b", "c", "a"]


def test_encode_labels_coerces_to_str() -> None:
    classes, encoded = encode_labels([1, 2, 1])
    assert classes.tolist() == ["1", "2"]
    assert encoded.tolist() == [0, 1, 0]


def test_balanced_class_weights_matches_sklearn_formula() -> None:
    encoded = np.array([0, 0, 0, 1], dtype=np.int64)
    weights = balanced_class_weights(encoded, 2)
    # n_samples / (n_classes * count): 4 / (2*3) and 4 / (2*1)
    assert weights == pytest.approx([4 / 6, 4 / 2])


def test_balanced_class_weights_guards_absent_class() -> None:
    weights = balanced_class_weights(np.array([0, 0], dtype=np.int64), 2)
    assert np.all(np.isfinite(weights))


def test_linear_warmup_factor_ramps_then_decays() -> None:
    total, warmup = 100, 10
    assert linear_warmup_factor(0, warmup, total) == 0.0
    assert linear_warmup_factor(5, warmup, total) == pytest.approx(0.5)
    assert linear_warmup_factor(warmup, warmup, total) == pytest.approx(1.0)
    assert linear_warmup_factor(total, warmup, total) == 0.0
    factors = [linear_warmup_factor(step, warmup, total) for step in range(total + 1)]
    assert max(factors) == pytest.approx(1.0)
    assert all(f >= 0.0 for f in factors)


def test_linear_warmup_factor_without_warmup_starts_at_peak() -> None:
    assert linear_warmup_factor(0, 0, 10) == pytest.approx(1.0)


def test_resolve_device_auto_detects_and_respects_explicit() -> None:
    fake_torch_gpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    fake_torch_cpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert resolve_device(None, fake_torch_gpu) == "cuda"
    assert resolve_device(None, fake_torch_cpu) == "cpu"
    assert resolve_device("cuda:1", fake_torch_cpu) == "cuda:1"


def test_optimizer_param_groups_excludes_bias_and_norm() -> None:
    weight = SimpleNamespace(requires_grad=True)
    bias = SimpleNamespace(requires_grad=True)
    norm_weight = SimpleNamespace(requires_grad=True)
    frozen = SimpleNamespace(requires_grad=False)
    model = SimpleNamespace(
        named_parameters=lambda: [
            ("encoder.layer.0.weight", weight),
            ("encoder.layer.0.bias", bias),
            ("encoder.LayerNorm.weight", norm_weight),
            ("embeddings.weight", frozen),
        ]
    )
    decayed, undecayed = optimizer_param_groups(model, 0.01)
    assert decayed["weight_decay"] == 0.01
    assert decayed["params"] == [weight]
    assert undecayed["weight_decay"] == 0.0
    assert undecayed["params"] == [bias, norm_weight]


def test_require_fitted_passes_when_attributes_present() -> None:
    fitted = SimpleNamespace(model_=object(), tokenizer_=object())
    require_fitted(fitted, "model_", "tokenizer_")  # must not raise
    with pytest.raises(TulipError):
        require_fitted(SimpleNamespace(model_=None), "model_")
