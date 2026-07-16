"""The shared torch training and batched-softmax inference loops.

torch is never imported here; the loops receive the already-imported module from
their callers, so this module stays importable without the optional extra.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = get_logger(__name__)


def linear_warmup_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate multiplier for linear warmup followed by linear decay.

    Args:
        step: Current optimizer step (0-based).
        warmup_steps: Steps spent ramping the rate from 0 to its peak.
        total_steps: Total optimizer steps over the whole run.

    Returns:
        A factor in ``[0, 1]`` to multiply the base learning rate by.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def resolve_device(device: str | None, torch_module: Any) -> str:
    """Resolve the compute device, auto-detecting CUDA when unspecified.

    Args:
        device: Explicit device string (e.g. ``"cpu"``, ``"cuda:0"``) or
            ``None`` for auto-detection.
        torch_module: The imported ``torch`` module (passed in so this stays
            testable without torch installed).

    Returns:
        The device string to use.
    """
    if device is not None:
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def optimizer_param_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """Split model parameters into decayed and non-decayed AdamW groups.

    Biases and normalisation parameters conventionally receive no weight
    decay; decaying them hurts transformer fine-tuning stability.

    Args:
        model: A ``torch.nn.Module``.
        weight_decay: Decay applied to the non-excluded group.

    Returns:
        Two AdamW parameter-group dicts (with and without weight decay).
    """
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if lowered.endswith(".bias") or "norm" in lowered:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_torch_classifier(
    torch: Any,
    model: Any,
    encode_batch: Callable[[np.ndarray], tuple[dict[str, Any], Any]],
    n_examples: int,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    seed: int,
    device: str,
    class_weights: np.ndarray | None = None,
) -> None:
    """Run a plain AdamW + linear-warmup training loop over a classifier head.

    Shared by the text and speech fine-tuning wrappers; the caller supplies
    ``encode_batch`` mapping an index array to ``(model inputs, target
    tensor)`` already placed on ``device``.

    Args:
        torch: The imported ``torch`` module.
        model: Model returning an object with a ``.logits`` attribute.
        encode_batch: Callable building one batch from example indices.
        n_examples: Number of training examples.
        epochs: Number of passes over the data.
        batch_size: Examples per forward pass.
        learning_rate: Peak AdamW learning rate.
        weight_decay: Decoupled weight decay (biases/norms excluded).
        warmup_ratio: Fraction of optimizer steps spent in linear warmup.
        gradient_accumulation_steps: Forward passes per optimizer step.
        max_grad_norm: Gradient clipping threshold (``<= 0`` disables it).
        seed: Seed for shuffling and torch RNG.
        device: Device the model lives on.
        class_weights: Optional per-class ``CrossEntropyLoss`` weights.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device)
    model.train()

    steps_per_epoch = math.ceil(n_examples / batch_size)
    optim_steps_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation_steps)
    total_steps = max(1, epochs * optim_steps_per_epoch)
    warmup_steps = round(warmup_ratio * total_steps)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(optimizer_param_groups(model, weight_decay), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: linear_warmup_factor(step, warmup_steps, total_steps)
    )

    for epoch in range(epochs):
        order = rng.permutation(n_examples)
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0
        for batch_index, start in enumerate(range(0, n_examples, batch_size)):
            indices = order[start : start + batch_size]
            inputs, targets = encode_batch(indices)
            logits = model(**inputs).logits
            loss = loss_fn(logits, targets)
            epoch_loss += float(loss.detach())
            n_batches += 1
            (loss / gradient_accumulation_steps).backward()
            is_last = start + batch_size >= n_examples
            if (batch_index + 1) % gradient_accumulation_steps == 0 or is_last:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        logger.info(
            "epoch %d/%d - mean training loss %.4f",
            epoch + 1,
            epochs,
            epoch_loss / max(1, n_batches),
        )
    model.eval()


def empty_proba(n_classes: int) -> np.ndarray:
    """Return the empty ``(0, n_classes)`` probability matrix for empty input.

    Every ``predict_proba`` short-circuits on an empty batch to this shape, so
    callers still get a well-formed two-dimensional array with the right column
    count to stack or index.

    Args:
        n_classes: Number of probability columns.

    Returns:
        A ``float64`` array of shape ``(0, n_classes)``.
    """
    return np.zeros((0, n_classes), dtype=np.float64)


def batched_softmax_probabilities(
    torch: Any,
    model: Any,
    inputs: Sequence[Any],
    encode_batch: Callable[[Sequence[Any]], Mapping[str, Any]],
    *,
    batch_size: int,
    n_classes: int,
    device: Any,
) -> np.ndarray:
    """Gradient-free batched inference shared by the fine-tuning wrappers.

    The wrappers differ only in how a batch of raw inputs becomes model
    tensors (tokenisation vs waveform feature extraction); the batching,
    device placement, forward pass, softmax, and stacking live here once.

    Args:
        torch: The imported ``torch`` module.
        model: Fitted model returning an object with a ``.logits`` attribute.
        inputs: Raw inputs (texts or audio paths).
        encode_batch: Maps one slice of ``inputs`` to the model's tensor kwargs
            (not yet on ``device``; placement happens here).
        batch_size: Inputs per forward pass.
        n_classes: Number of probability columns (for the empty-input case).
        device: Device the fitted model lives on.

    Returns:
        Array of shape ``(len(inputs), n_classes)`` of softmax probabilities.
    """
    items = list(inputs)
    if not items:
        return empty_proba(n_classes)
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            encoded = encode_batch(items[start : start + batch_size])
            tensors = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**tensors).logits
            probabilities = torch.softmax(logits, dim=-1)
            rows.append(probabilities.detach().cpu().numpy())
    return np.vstack(rows).astype(np.float64)
