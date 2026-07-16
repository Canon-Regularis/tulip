"""Extras-gated numeric tests for the shared torch training/inference loops.

The rest of the neural test-suite exercises only failure paths and plumbing with
a faked torch, so the real forward/backward/optimizer/softmax numerics never run
in a torch-less CI job. These tests run the genuine loops
(:func:`~tulip.models._torch_loops.train_torch_classifier` and
:func:`~tulip.models._torch_loops.batched_softmax_probabilities`, the ones the
text and speech fine-tuning wrappers share) against a tiny real ``nn.Linear``
head. They need only torch (no transformers, no download, no network), so they
run wherever the ``transformers`` or ``speech`` extra is installed and skip
elsewhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tulip.models._torch_loops import (  # noqa: E402  (import after the skip guard)
    batched_softmax_probabilities,
    optimizer_param_groups,
    resolve_device,
    train_torch_classifier,
)


class _LinearHead(torch.nn.Module):
    """A minimal classifier: one linear layer returning an object with .logits."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_classes)
        self.layernorm = torch.nn.LayerNorm(n_features)

    def forward(self, x):
        return SimpleNamespace(logits=self.linear(self.layernorm(x)))


def _separable_dataset() -> tuple[np.ndarray, np.ndarray]:
    """A trivially linearly separable two-class dataset."""
    class0 = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [1.0, 0.1]], dtype=np.float32)
    class1 = np.array([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.1, 1.0]], dtype=np.float32)
    features = np.vstack([class0, class1])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return features, labels


def test_training_loop_learns_a_separable_task() -> None:
    features, labels = _separable_dataset()
    model = _LinearHead(2, 2)

    def encode_batch(indices: np.ndarray) -> tuple[dict, object]:
        return (
            {"x": torch.tensor(features[indices], dtype=torch.float32)},
            torch.tensor(labels[indices], dtype=torch.long),
        )

    train_torch_classifier(
        torch,
        model,
        encode_batch,
        n_examples=len(labels),
        epochs=60,
        batch_size=4,
        learning_rate=0.1,
        weight_decay=0.0,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        seed=0,
        device="cpu",
    )
    assert model.training is False  # the loop leaves the model in eval mode

    def encode_predict(items):
        return {"x": torch.tensor(np.array(items), dtype=torch.float32)}

    proba = batched_softmax_probabilities(
        torch, model, list(features), encode_predict, batch_size=4, n_classes=2, device="cpu"
    )
    assert proba.shape == (8, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)  # real softmax rows normalise
    # The loop actually learned the separable task, not just ran without error.
    assert (proba.argmax(axis=1) == labels).mean() >= 0.75


def test_batched_softmax_handles_empty_input() -> None:
    model = _LinearHead(2, 2)

    def encode_predict(items):
        return {"x": torch.tensor(np.array(items), dtype=torch.float32)}

    proba = batched_softmax_probabilities(
        torch, model, [], encode_predict, batch_size=4, n_classes=2, device="cpu"
    )
    assert proba.shape == (0, 2)


def test_optimizer_param_groups_excludes_bias_and_norm_on_a_real_module() -> None:
    model = _LinearHead(4, 3)
    decay, no_decay = optimizer_param_groups(model, weight_decay=0.01)
    assert decay["weight_decay"] == 0.01
    assert no_decay["weight_decay"] == 0.0
    # linear.weight decays; linear.bias, layernorm.weight, layernorm.bias do not.
    assert len(decay["params"]) == 1
    assert len(no_decay["params"]) == 3


def test_resolve_device_with_real_torch() -> None:
    resolved = resolve_device(None, torch)
    assert resolved in {"cpu", "cuda"}
    assert resolve_device("cpu", torch) == "cpu"
