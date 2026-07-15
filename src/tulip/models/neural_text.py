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
training loop short enough to audit. The loop itself — and every other piece
of machinery shared with the speech and fastText wrappers — lives in
:mod:`tulip.models._common`.

All heavy dependencies (torch, transformers) are imported lazily inside
methods via :func:`tulip.utils.optional.optional_import`; importing this
module never requires an optional dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.base import BaseEstimator, ClassifierMixin

from tulip.core.exceptions import ConfigurationError
from tulip.models._common import (
    ArgmaxPredictMixin,
    balanced_class_weights,
    batched_softmax_probabilities,
    checkpoint_factory,
    require_fitted,
    resolve_device,
    train_torch_classifier,
    validate_common_training_params,
    validate_fit_inputs,
)
from tulip.models.registry import MODELS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

logger = get_logger(__name__)

#: Registry name -> Hugging Face checkpoint for the built-in text models.
TEXT_CHECKPOINTS: dict[str, str] = {
    "herbert": "allegro/herbert-base-cased",
    "polish_roberta": "sdadas/polish-roberta-base-v2",
    "mbert": "bert-base-multilingual-cased",
    "xlm_roberta": "xlm-roberta-base",
}

__all__ = ["TEXT_CHECKPOINTS", "TransformerTextClassifier"]


class TransformerTextClassifier(ArgmaxPredictMixin, ClassifierMixin, BaseEstimator):
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
        class_weight: str | None = "balanced",
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        """Configure the wrapper; no heavy dependency is imported here.

        Args:
            checkpoint: Hugging Face model id or local path.
            max_length: Token-truncation length for training and inference.
            epochs: Training epochs.
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

        Note:
            Per the scikit-learn estimator contract, ``__init__`` only stores
            parameters; validation happens in :meth:`fit` so values injected
            via ``set_params`` (e.g. by ``GridSearchCV``) are validated too.
        """
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

    def _validate_hyperparameters(self) -> None:
        """Validate constructor/set_params values (called from :meth:`fit`)."""
        if self.max_length < 1:
            raise ConfigurationError(f"max_length must be >= 1, got {self.max_length}")
        validate_common_training_params(self)

    def fit(self, X: Sequence[str], y: Sequence[Any]) -> TransformerTextClassifier:
        """Fine-tune the checkpoint on raw texts.

        Args:
            X: Sequence of raw documents.
            y: Parallel sequence of labels (any hashables; coerced to str).

        Returns:
            ``self``, fitted.

        Raises:
            ConfigurationError: if a hyperparameter is out of range.
            MissingDependencyError: if torch/transformers are not installed.
            DataError: if inputs are empty, mismatched, or single-class.
        """
        self._validate_hyperparameters()  # before imports: valid config first
        torch = optional_import(
            "torch", extra="transformers", purpose="fine-tuning transformer text models"
        )
        transformers = optional_import(
            "transformers", extra="transformers", purpose="transformer text models"
        )
        texts = [str(text) for text in X]
        classes, encoded = validate_fit_inputs(texts, y)

        device = resolve_device(self.device, torch)
        # Seed BEFORE model construction: from_pretrained randomly initialises
        # the new classification head from the ambient RNG, so seeding only
        # inside the training loop would leave fit() non-reproducible.
        torch.manual_seed(self.seed)
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
            inputs = {
                key: value.to(device)
                for key, value in self._tokenize(tokenizer, [texts[i] for i in indices]).items()
            }
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

    def _tokenize(self, tokenizer: Any, texts: Sequence[str]) -> Any:
        """Tokenise one batch with the wrapper's truncation settings."""
        return tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

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
        return batched_softmax_probabilities(
            torch,
            self.model_,
            [str(text) for text in X],
            lambda batch: self._tokenize(self.tokenizer_, batch),
            batch_size=self.batch_size,
            n_classes=len(self.classes_),
            device=self.device_,
        )


def _register_factories() -> None:
    for name, checkpoint in TEXT_CHECKPOINTS.items():
        MODELS.add(
            name,
            checkpoint_factory(TransformerTextClassifier, checkpoint),
            # training_aware: the constructor accepts the shared TrainingConfig
            # knobs (batch_size/epochs/learning_rate), so the experiment runner
            # may merge them into the model params.
            # raw_input: tokenises raw texts itself, so it needs no feature
            # extractors -- DialectClassifier checks this before fitting.
            # extra: torch + transformers, installed by the `transformers` extra.
            metadata={"training_aware": True, "raw_input": True, "extra": "transformers"},
        )


_register_factories()
