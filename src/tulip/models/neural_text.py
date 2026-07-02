"""Transformer text classifiers fine-tuned for Polish dialect detection.

:class:`TransformerTextClassifier` wraps Hugging Face
``AutoModelForSequenceClassification`` + ``AutoTokenizer`` in the scikit-learn
estimator API (``fit`` / ``predict`` / ``predict_proba`` / ``classes_``) so
transformer fine-tuning composes with the rest of tulip exactly like a
classical model.

Design note: training uses a plain torch loop (``AdamW`` with decoupled weight
decay, linear warmup + linear decay schedule, ``CrossEntropyLoss`` with
optional balanced class weights) rather than the Hugging Face ``Trainer``.
This keeps the dependency surface small (no ``accelerate`` requirement at
runtime), makes seeding and class weighting fully explicit, and keeps the
training loop short enough to audit.

All heavy dependencies (torch, transformers) are imported lazily inside
methods via :func:`tulip.utils.optional.optional_import`; importing this
module never requires an optional dependency. This module also hosts small
shared helpers (label encoding, warmup schedule, device resolution, the
generic torch training loop) reused by :mod:`tulip.models.neural_audio`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tulip.core.exceptions import ConfigurationError, DataError, TulipError
from tulip.models.registry import MODELS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

logger = get_logger(__name__)

#: Registry name -> Hugging Face checkpoint for the built-in text models.
TEXT_CHECKPOINTS: dict[str, str] = {
    "herbert": "allegro/herbert-base-cased",
    "polish_roberta": "sdadas/polish-roberta-base-v2",
    "mbert": "bert-base-multilingual-cased",
    "xlm_roberta": "xlm-roberta-base",
}

__all__ = [
    "TEXT_CHECKPOINTS",
    "TransformerTextClassifier",
    "balanced_class_weights",
    "checkpoint_factory",
    "encode_labels",
    "linear_warmup_factor",
    "optimizer_param_groups",
    "require_fitted",
    "resolve_device",
    "train_torch_classifier",
]


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


def checkpoint_factory(cls: type, checkpoint: str) -> Callable[..., Any]:
    """Build a registry factory that pre-binds a default checkpoint.

    The bound checkpoint (and every other constructor parameter) remains
    overridable through the factory's keyword arguments, so experiment configs
    can swap checkpoints without registering new names.

    Args:
        cls: The wrapper class to instantiate.
        checkpoint: Default Hugging Face checkpoint or model source.

    Returns:
        A keyword-only factory suitable for ``Registry.add``.
    """

    def factory(**params: Any) -> Any:
        params.setdefault("checkpoint", checkpoint)
        return cls(**params)

    safe = checkpoint.replace("/", "_").replace("-", "_").replace(".", "_")
    factory.__name__ = f"make_{cls.__name__.lower()}_{safe}"
    factory.__qualname__ = factory.__name__
    factory.__doc__ = f"Create a :class:`{cls.__name__}` pre-bound to checkpoint {checkpoint!r}."
    return factory


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


class TransformerTextClassifier(ClassifierMixin, BaseEstimator):
    """Fine-tune a Hugging Face sequence-classification model on raw texts.

    Follows scikit-learn conventions: ``fit(texts, y)``, ``predict``,
    ``predict_proba`` (softmax over logits, batched, under ``no_grad``), and a
    ``classes_`` attribute after fitting. The fitted ``model_`` and
    ``tokenizer_`` attributes are public so the explainability module can read
    attention weights and token alignments.

    Attributes:
        classes_: Sorted array of class labels (after ``fit``).
        model_: The fine-tuned Hugging Face model in eval mode (after ``fit``).
        tokenizer_: The matching tokenizer (after ``fit``).
        device_: Device the fitted model lives on (after ``fit``).
    """

    def __init__(
        self,
        checkpoint: str = "allegro/herbert-base-cased",
        *,
        max_length: int = 256,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        class_weight: str | None = None,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        """Configure the wrapper; no heavy dependency is imported here.

        Args:
            checkpoint: Hugging Face model checkpoint to fine-tune.
            max_length: Token truncation length (padding is per-batch dynamic).
            epochs: Training epochs (default per tulip ``TrainingConfig``).
            batch_size: Batch size for training and inference.
            learning_rate: Peak AdamW learning rate.
            weight_decay: Decoupled weight decay (biases/norms excluded).
            warmup_ratio: Fraction of optimizer steps used for linear warmup.
            gradient_accumulation_steps: Forward passes per optimizer step,
                for large effective batches on small GPUs.
            max_grad_norm: Gradient clipping threshold (``<= 0`` disables).
            class_weight: ``"balanced"`` to reweight the loss by inverse class
                frequency, or ``None``.
            device: Explicit device, or ``None`` to use CUDA when available.
            seed: Seed controlling shuffling and torch initialisation.

        Raises:
            ConfigurationError: if a hyperparameter is out of range.
        """
        if max_length < 1:
            raise ConfigurationError(f"max_length must be >= 1, got {max_length}")
        if epochs < 1:
            raise ConfigurationError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ConfigurationError(f"batch_size must be >= 1, got {batch_size}")
        if learning_rate <= 0:
            raise ConfigurationError(f"learning_rate must be > 0, got {learning_rate}")
        if not 0.0 <= warmup_ratio <= 1.0:
            raise ConfigurationError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")
        if gradient_accumulation_steps < 1:
            raise ConfigurationError(
                f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}"
            )
        if class_weight not in (None, "balanced"):
            raise ConfigurationError(
                f'class_weight must be None or "balanced", got {class_weight!r}'
            )
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.class_weight = class_weight
        self.device = device
        self.seed = seed

    def fit(self, X: Sequence[str], y: Sequence[Any]) -> TransformerTextClassifier:
        """Fine-tune the checkpoint on raw texts.

        Args:
            X: Sequence of raw documents.
            y: Parallel sequence of labels (any hashables; coerced to str).

        Returns:
            ``self``, fitted.

        Raises:
            MissingDependencyError: if torch/transformers are not installed.
            DataError: if inputs are empty, mismatched, or single-class.
        """
        torch = optional_import(
            "torch", extra="transformers", purpose="fine-tuning transformer text models"
        )
        transformers = optional_import(
            "transformers", extra="transformers", purpose="transformer text models"
        )
        texts = [str(text) for text in X]
        if not texts:
            raise DataError("cannot fit on an empty dataset")
        if len(texts) != len(y):
            raise DataError(f"X and y length mismatch: {len(texts)} != {len(y)}")
        classes, encoded = encode_labels(y)
        if len(classes) < 2:
            raise DataError(f"need at least 2 classes to fit, got {len(classes)}")

        device = resolve_device(self.device, torch)
        id2label = {index: str(label) for index, label in enumerate(classes)}
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.checkpoint)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            num_labels=len(classes),
            id2label=id2label,
            label2id={label: index for index, label in id2label.items()},
        )
        logger.info(
            "fine-tuning %s on %d texts / %d classes (device=%s)",
            self.checkpoint,
            len(texts),
            len(classes),
            device,
        )

        def encode_batch(indices: np.ndarray) -> tuple[dict[str, Any], Any]:
            batch_texts = [texts[i] for i in indices]
            encoding = tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in encoding.items()}
            targets = torch.as_tensor(encoded[indices], dtype=torch.long, device=device)
            return inputs, targets

        class_weights = None
        if self.class_weight == "balanced":
            class_weights = balanced_class_weights(encoded, len(classes))
        train_torch_classifier(
            torch,
            model,
            encode_batch,
            len(texts),
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
            seed=self.seed,
            device=device,
            class_weights=class_weights,
        )
        self.classes_ = classes
        self.model_ = model
        self.tokenizer_ = tokenizer
        self.device_ = device
        return self

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Return softmax class probabilities, batched and gradient-free.

        Args:
            X: Sequence of raw documents.

        Returns:
            Array of shape ``(len(X), n_classes)``; columns follow ``classes_``.

        Raises:
            TulipError: if the model has not been fitted.
            MissingDependencyError: if torch/transformers are not installed.
        """
        require_fitted(self, "model_", "tokenizer_")
        torch = optional_import("torch", extra="transformers", purpose="transformer text inference")
        texts = [str(text) for text in X]
        if not texts:
            return np.zeros((0, len(self.classes_)), dtype=np.float64)
        self.model_.eval()
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoding = self.tokenizer_(
                    batch,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device_) for key, value in encoding.items()}
                logits = self.model_(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)
                rows.append(probabilities.detach().cpu().numpy())
        return np.vstack(rows).astype(np.float64)

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Return the most probable class label for each text."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def _register_factories() -> None:
    for name, checkpoint in TEXT_CHECKPOINTS.items():
        MODELS.add(name, checkpoint_factory(TransformerTextClassifier, checkpoint))


_register_factories()
