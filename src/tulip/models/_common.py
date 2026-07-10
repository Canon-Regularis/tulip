"""Shared estimator machinery for the neural and fastText model wrappers.

Everything the transformer-text, speech, and fastText wrappers have in common
lives here exactly once: label encoding, fit-input validation, the common
hyperparameter checks, seed-spelling reconciliation, checkpoint-bound registry
factories, the argmax ``predict`` mixin, and the two torch loops (training
and batched softmax inference).

The module is import-cheap: torch is never imported here — the torch loops
receive the already-imported module from their callers, which keeps every
heavy dependency lazy and this module fully importable (and its pure-Python
helpers fully testable) without any optional extra installed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from tulip.core.exceptions import ConfigurationError, DataError, TulipError
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = get_logger(__name__)

__all__ = [
    "ArgmaxPredictMixin",
    "balanced_class_weights",
    "batched_softmax_probabilities",
    "checkpoint_factory",
    "encode_labels",
    "linear_warmup_factor",
    "optimizer_param_groups",
    "reconcile_seed_param",
    "require_fitted",
    "resolve_device",
    "train_torch_classifier",
    "validate_common_training_params",
    "validate_fit_inputs",
]


# --------------------------------------------------------------------- labels


def encode_labels(y: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Encode arbitrary labels as integer ids over a sorted class vocabulary.

    Args:
        y: Sequence of labels (coerced to ``str``).

    Returns:
        ``(classes, encoded)`` where ``classes`` is the sorted array of unique
        label strings and ``encoded`` is an ``int64`` array with
        ``classes[encoded[i]] == str(y[i])``.
    """
    labels = np.asarray([str(value) for value in y], dtype=object)
    classes, encoded = np.unique(labels, return_inverse=True)
    return classes, encoded.astype(np.int64)


def validate_fit_inputs(inputs: Sequence[Any], y: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Validate the universal ``fit(X, y)`` preconditions and encode ``y``.

    Every wrapper shares the same three requirements: a non-empty ``X``,
    aligned lengths, and at least two classes.

    Args:
        inputs: The raw model inputs (texts or audio paths), already coerced.
        y: Parallel label sequence.

    Returns:
        ``(classes, encoded)`` as from :func:`encode_labels`.

    Raises:
        DataError: if inputs are empty, mismatched, or single-class.
    """
    if not inputs:
        raise DataError("cannot fit on an empty dataset")
    if len(inputs) != len(y):
        raise DataError(f"X and y length mismatch: {len(inputs)} != {len(y)}")
    classes, encoded = encode_labels(y)
    if len(classes) < 2:
        raise DataError(f"need at least 2 classes to fit, got {len(classes)}")
    return classes, encoded


def balanced_class_weights(encoded: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute sklearn-style ``"balanced"`` class weights from encoded labels.

    Uses ``n_samples / (n_classes * count_per_class)`` so rare dialects are
    not drowned out by majority classes during fine-tuning.

    Args:
        encoded: Integer label ids in ``[0, n_classes)``.
        n_classes: Total number of classes.

    Returns:
        A ``float64`` array of per-class weights of length ``n_classes``.
    """
    counts = np.bincount(np.asarray(encoded, dtype=np.int64), minlength=n_classes)
    counts = np.maximum(counts, 1)  # guard classes absent from this fold
    return len(encoded) / (n_classes * counts.astype(np.float64))


# ----------------------------------------------------------------- estimators


def require_fitted(estimator: Any, *attributes: str) -> None:
    """Raise if ``estimator`` has not been fitted yet.

    Args:
        estimator: The wrapper instance to check.
        attributes: Post-fit attribute names that must be present.

    Raises:
        TulipError: if any required attribute is missing.
    """
    for attribute in attributes:
        if getattr(estimator, attribute, None) is None:
            raise TulipError(
                f"{type(estimator).__name__} is not fitted yet; call fit(X, y) before predicting"
            )


def validate_common_training_params(estimator: Any) -> None:
    """Validate the training knobs shared by every fine-tuning wrapper.

    Called from each wrapper's ``fit`` (never ``__init__``: the sklearn
    estimator contract requires ``set_params``-injected values to be
    validated too). Wrappers add their modality-specific checks on top.

    Raises:
        ConfigurationError: if a hyperparameter is out of range.
    """
    if estimator.epochs < 1:
        raise ConfigurationError(f"epochs must be >= 1, got {estimator.epochs}")
    if estimator.batch_size < 1:
        raise ConfigurationError(f"batch_size must be >= 1, got {estimator.batch_size}")
    if estimator.learning_rate <= 0:
        raise ConfigurationError(f"learning_rate must be > 0, got {estimator.learning_rate}")
    if not 0.0 <= estimator.warmup_ratio <= 1.0:
        raise ConfigurationError(f"warmup_ratio must be in [0, 1], got {estimator.warmup_ratio}")
    if estimator.gradient_accumulation_steps < 1:
        raise ConfigurationError(
            f"gradient_accumulation_steps must be >= 1, got {estimator.gradient_accumulation_steps}"
        )
    if estimator.class_weight not in (None, "balanced"):
        raise ConfigurationError(
            f'class_weight must be None or "balanced", got {estimator.class_weight!r}'
        )


def reconcile_seed_param(params: dict[str, Any]) -> None:
    """Map scikit-learn's ``random_state`` spelling onto the wrappers' ``seed``.

    The classical factories accept both spellings (see
    ``tulip.models.classical``); this keeps the neural/fastText factories
    interchangeable with them in experiment configs. Mutates ``params``.

    Raises:
        ConfigurationError: if ``random_state`` and ``seed`` disagree.
    """
    if "random_state" not in params:
        return
    random_state = params.pop("random_state")
    if "seed" in params and params["seed"] != random_state:
        raise ConfigurationError(
            f"conflicting seeds: random_state={random_state!r} vs seed={params['seed']!r}"
        )
    params.setdefault("seed", random_state)


def checkpoint_factory(cls: type, checkpoint: str) -> Callable[..., Any]:
    """Build a registry factory that pre-binds a default checkpoint.

    The bound checkpoint (and every other constructor parameter) remains
    overridable through the factory's keyword arguments, so experiment configs
    can swap checkpoints without registering new names. ``random_state`` is
    accepted as an alias for ``seed`` (scikit-learn spelling).

    Args:
        cls: The wrapper class to instantiate.
        checkpoint: Default Hugging Face checkpoint or model source.

    Returns:
        A keyword-only factory suitable for ``Registry.add``.
    """

    def factory(**params: Any) -> Any:
        params.setdefault("checkpoint", checkpoint)
        reconcile_seed_param(params)
        return cls(**params)

    safe = checkpoint.replace("/", "_").replace("-", "_").replace(".", "_")
    factory.__name__ = f"make_{cls.__name__.lower()}_{safe}"
    factory.__qualname__ = factory.__name__
    factory.__doc__ = f"Create a :class:`{cls.__name__}` pre-bound to checkpoint {checkpoint!r}."
    return factory


class ArgmaxPredictMixin:
    """Derive ``predict`` from ``predict_proba`` (argmax over ``classes_``).

    Guarantees ``predict == classes_[argmax(predict_proba)]`` by construction
    for every wrapper that mixes this in, instead of four hand-maintained
    copies of the same line.
    """

    classes_: np.ndarray

    if TYPE_CHECKING:  # concrete classes provide the real implementation

        def predict_proba(self, X: Sequence[Any]) -> np.ndarray: ...

    def predict(self, X: Sequence[Any]) -> np.ndarray:
        """Return the most probable class label for each input."""
        probabilities = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(probabilities, axis=1)]


# ---------------------------------------------------------------- torch loops


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
        return np.zeros((0, n_classes), dtype=np.float64)
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
