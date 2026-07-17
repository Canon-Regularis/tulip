"""End-to-end fine-tuned Hugging Face audio-classification speech models.

:class:`FinetunedSpeechClassifier` fine-tunes a Hugging Face audio
classification model end-to-end: wav2vec2 and HuBERT via
``AutoModelForAudioClassification``, Whisper via
``WhisperForAudioClassification`` (encoder + classification head). The frozen
embedding + head alternative lives in
:mod:`tulip.models.neural_audio_embedding`; the two share
:mod:`tulip.models._audio_common`.

It follows scikit-learn conventions (``fit(paths, y)`` / ``predict`` /
``predict_proba`` / ``classes_``) over sequences of audio file paths, decoding
through the canonical shared loader. Fine-tuning reuses the shared torch
training loop from :mod:`tulip.models._torch_loops` (AdamW + linear warmup,
optional balanced class weights). All heavy dependencies (torch, transformers,
soundfile) are
imported lazily inside methods, so importing this module never requires an
optional dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sklearn.base import BaseEstimator, ClassifierMixin

from tulip.core.exceptions import ConfigurationError
from tulip.models._audio_common import (
    TARGET_SAMPLE_RATE,
    is_whisper_extractor,
    load_clipped_waveforms,
)
from tulip.models._common import (
    ArgmaxPredictMixin,
    batched_softmax_probabilities,
    checkpoint_factory,
    label_id_maps,
    require_fitted,
    resolve_class_weights,
    resolve_device,
    train_classifier_from_estimator,
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

#: Registry name -> Hugging Face checkpoint for the fine-tuned speech models.
FINETUNE_CHECKPOINTS: dict[str, str] = {
    "wav2vec2": "facebook/wav2vec2-xls-r-300m",
    "hubert": "facebook/hubert-base-ls960",
    "whisper": "openai/whisper-small",
}

__all__ = ["FINETUNE_CHECKPOINTS", "FinetunedSpeechClassifier", "SpeechClassifier"]


class FinetunedSpeechClassifier(ArgmaxPredictMixin, ClassifierMixin, BaseEstimator):
    """Fine-tune a Hugging Face audio-classification model on audio files.

    Follows scikit-learn conventions: ``fit(paths, y)``, ``predict``,
    ``predict_proba`` (softmax over logits, batched, under ``no_grad``), and a
    ``classes_`` attribute after fitting. Waveforms are decoded per batch
    (never all at once), keeping memory bounded on large corpora at the cost
    of re-decoding each epoch; decoding is cheap next to a transformer
    forward/backward pass.

    Whisper checkpoints are detected from their feature extractor and loaded
    through ``WhisperForAudioClassification``; everything else goes through
    ``AutoModelForAudioClassification``.

    Attributes:
        classes_: Sorted array of class labels (after ``fit``).
        model_: The fine-tuned Hugging Face model in eval mode (after ``fit``).
        feature_extractor_: The matching feature extractor (after ``fit``).
        device_: Device the fitted model lives on (after ``fit``).
    """

    def __init__(
        self,
        checkpoint: str = "facebook/wav2vec2-xls-r-300m",
        *,
        max_seconds: float = 10.0,
        sample_rate: int = TARGET_SAMPLE_RATE,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        class_weight: str | None = None,
        freeze_feature_encoder: bool = True,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        """Configure the wrapper; no heavy dependency is imported here.

        The default batch size (8) is smaller than tulip's ``TrainingConfig``
        default because 16 kHz waveforms are orders of magnitude longer than
        256 text tokens; raise ``gradient_accumulation_steps`` to recover a
        larger effective batch on small GPUs.

        Args:
            checkpoint: Hugging Face audio-classification checkpoint.
            max_seconds: Clips are truncated to this many seconds.
            sample_rate: Model input sample rate in Hz.
            epochs: Training epochs.
            batch_size: Batch size for training and inference.
            learning_rate: Peak AdamW learning rate.
            weight_decay: Decoupled weight decay (biases/norms excluded).
            warmup_ratio: Fraction of optimizer steps used for linear warmup.
            gradient_accumulation_steps: Forward passes per optimizer step.
            max_grad_norm: Gradient clipping threshold (``<= 0`` disables).
            class_weight: ``"balanced"`` to reweight the loss by inverse class
                frequency, or ``None``.
            freeze_feature_encoder: Freeze the CNN waveform encoder when the
                model supports it (standard for stable wav2vec2/HuBERT
                fine-tuning); silently ignored otherwise.
            device: Explicit device, or ``None`` to use CUDA when available.
            seed: Seed controlling shuffling and torch initialisation.

        Note:
            Per the scikit-learn estimator contract, ``__init__`` only stores
            parameters; validation happens in :meth:`fit` so values injected
            via ``set_params`` (e.g. by ``GridSearchCV``) are validated too.
        """
        self.checkpoint = checkpoint
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.class_weight = class_weight
        self.freeze_feature_encoder = freeze_feature_encoder
        self.device = device
        self.seed = seed

    def _validate_hyperparameters(self) -> None:
        """Validate constructor/set_params values (called from :meth:`fit`)."""
        if self.max_seconds <= 0:
            raise ConfigurationError(f"max_seconds must be > 0, got {self.max_seconds}")
        if self.sample_rate < 1:
            raise ConfigurationError(f"sample_rate must be >= 1, got {self.sample_rate}")
        validate_common_training_params(self)

    def _encode_features(self, feature_extractor: Any, waveforms: list[np.ndarray]) -> Any:
        """Run the feature extractor with padding appropriate to its family."""
        if is_whisper_extractor(feature_extractor):
            # Whisper pads/truncates to its fixed 30 s window internally.
            return feature_extractor(waveforms, sampling_rate=self.sample_rate, return_tensors="pt")
        return feature_extractor(
            waveforms, sampling_rate=self.sample_rate, padding=True, return_tensors="pt"
        )

    def fit(self, X: Sequence[str | Path], y: Sequence[Any]) -> FinetunedSpeechClassifier:
        """Fine-tune the checkpoint on labelled audio files.

        Args:
            X: Sequence of audio file paths.
            y: Parallel sequence of labels (any hashables; coerced to str).

        Returns:
            ``self``, fitted.

        Raises:
            ConfigurationError: if a hyperparameter is out of range.
            MissingDependencyError: if torch/transformers are not installed.
            DataError: if inputs are empty, mismatched, single-class, or
                undecodable.
        """
        self._validate_hyperparameters()  # before imports: valid config first
        torch = optional_import("torch", extra="speech", purpose="fine-tuning speech models")
        transformers = optional_import(
            "transformers", extra="speech", purpose="speech classification models"
        )
        paths = [Path(path) for path in X]
        classes, encoded = validate_fit_inputs(paths, y)

        device = resolve_device(self.device, torch)
        # Seed BEFORE model construction: from_pretrained randomly initialises
        # the new classification head from the ambient RNG, so seeding only
        # inside the training loop would leave fit() non-reproducible.
        torch.manual_seed(self.seed)
        feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(self.checkpoint)
        id2label, label2id = label_id_maps(classes)
        model_cls = (
            transformers.WhisperForAudioClassification
            if is_whisper_extractor(feature_extractor)
            else transformers.AutoModelForAudioClassification
        )
        model = model_cls.from_pretrained(
            self.checkpoint,
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id,
        )
        if self.freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        logger.info(
            "fine-tuning %s on %d audio files / %d classes (device=%s)",
            self.checkpoint,
            len(paths),
            len(classes),
            device,
        )

        def encode_batch(indices: np.ndarray) -> tuple[dict[str, Any], Any]:
            batch_paths = [paths[i] for i in indices]
            waveforms = load_clipped_waveforms(
                batch_paths, sample_rate=self.sample_rate, max_seconds=self.max_seconds
            )
            features = self._encode_features(feature_extractor, waveforms)
            inputs = {key: value.to(device) for key, value in features.items()}
            targets = torch.as_tensor(encoded[indices], dtype=torch.long, device=device)
            return inputs, targets

        class_weights = resolve_class_weights(self, encoded, len(classes))
        train_classifier_from_estimator(
            self, torch, model, encode_batch, len(paths), device=device, class_weights=class_weights
        )
        self.classes_ = classes
        self.model_ = model
        self.feature_extractor_ = feature_extractor
        self.device_ = device
        return self

    def predict_proba(self, X: Sequence[str | Path]) -> np.ndarray:
        """Return softmax class probabilities, batched and gradient-free.

        Args:
            X: Sequence of audio file paths.

        Returns:
            Array of shape ``(len(X), n_classes)``; columns follow ``classes_``.

        Raises:
            TulipError: if the model has not been fitted.
            MissingDependencyError: if torch is not installed.
        """
        require_fitted(self, "model_", "feature_extractor_")
        torch = optional_import("torch", extra="speech", purpose="speech model inference")

        def encode_batch(batch_paths: Sequence[Path]) -> Any:
            waveforms = load_clipped_waveforms(
                list(batch_paths), sample_rate=self.sample_rate, max_seconds=self.max_seconds
            )
            return self._encode_features(self.feature_extractor_, waveforms)

        return batched_softmax_probabilities(
            torch,
            self.model_,
            [Path(path) for path in X],
            encode_batch,
            batch_size=self.batch_size,
            n_classes=len(self.classes_),
            device=self.device_,
        )


#: Alias matching the architecture document's family name for these wrappers.
SpeechClassifier = FinetunedSpeechClassifier


def _register_factories() -> None:
    for name, checkpoint in FINETUNE_CHECKPOINTS.items():
        MODELS.add(
            name,
            checkpoint_factory(FinetunedSpeechClassifier, checkpoint),
            # training_aware: accepts the shared TrainingConfig knobs; the
            # embedding models do not (frozen encoder + sklearn head).
            # raw_input: decodes audio paths itself, so it needs no extractors.
            # extra: torch + transformers, from the `speech` extra.
            metadata={"training_aware": True, "raw_input": True, "extra": "speech"},
        )


_register_factories()
