"""Neural speech classifiers for dialect identification from raw audio.

Two complementary strategies are provided:

* :class:`FinetunedSpeechClassifier` fine-tunes a Hugging Face audio
  classification model end-to-end: wav2vec2 and HuBERT via
  ``AutoModelForAudioClassification``, Whisper via
  ``WhisperForAudioClassification`` (encoder + classification head).
* :class:`EmbeddingSpeechClassifier` extracts frozen pretrained speaker
  embeddings (speechbrain ECAPA-TDNN / x-vector) and fits a scikit-learn
  ``LogisticRegression`` head on top. This is embedding + head, **not**
  end-to-end fine-tuning: the encoder weights never change, which is far
  cheaper and works well on small dialect corpora.

Both wrappers follow scikit-learn conventions (``fit(paths, y)`` /
``predict`` / ``predict_proba`` / ``classes_``) over sequences of audio file
paths. Decoding prefers the shared loader in ``tulip.features.audio.loading``
when importable and falls back to soundfile + scipy polyphase resampling;
either way model input is mono float32 at 16 kHz.

Fine-tuning reuses the plain torch loop from :mod:`tulip.models.neural_text`
(AdamW + linear warmup, optional balanced class weights) — see that module
for why tulip avoids the Hugging Face ``Trainer``. All heavy dependencies
(torch, transformers, speechbrain, soundfile) are imported lazily inside
methods via :func:`tulip.utils.optional.optional_import`; importing this
module never requires an optional dependency.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly
from sklearn.base import BaseEstimator, ClassifierMixin

from tulip.core.exceptions import ConfigurationError, DataError, TulipError
from tulip.models.neural_text import (
    balanced_class_weights,
    checkpoint_factory,
    encode_labels,
    require_fitted,
    resolve_device,
    train_torch_classifier,
)
from tulip.models.registry import MODELS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

logger = get_logger(__name__)

#: Sample rate every model input is resampled to (Hz).
TARGET_SAMPLE_RATE = 16_000

#: Registry name -> Hugging Face checkpoint for the fine-tuned speech models.
FINETUNE_CHECKPOINTS: dict[str, str] = {
    "wav2vec2": "facebook/wav2vec2-xls-r-300m",
    "hubert": "facebook/hubert-base-ls960",
    "whisper": "openai/whisper-small",
}

#: Registry name -> speechbrain source for the embedding + head models.
EMBEDDING_CHECKPOINTS: dict[str, str] = {
    "ecapa_tdnn": "speechbrain/spkrec-ecapa-voxceleb",
    "xvector": "speechbrain/spkrec-xvect-voxceleb",
}

#: Function names probed on ``tulip.features.audio.loading`` for the shared decoder.
_SHARED_LOADER_CANDIDATES = ("load_audio", "load_waveform", "load")

__all__ = [
    "EMBEDDING_CHECKPOINTS",
    "FINETUNE_CHECKPOINTS",
    "TARGET_SAMPLE_RATE",
    "EmbeddingSpeechClassifier",
    "FinetunedSpeechClassifier",
    "SpeechClassifier",
    "ensure_mono",
    "is_whisper_extractor",
    "load_waveform",
    "normalise_loader_output",
    "resample_waveform",
]


def ensure_mono(waveform: Any) -> np.ndarray:
    """Coerce a decoded waveform to a 1-D mono float32 array.

    Multi-channel input is averaged over the channel axis. Decoders disagree
    on layout (soundfile emits ``(frames, channels)``, torchaudio emits
    ``(channels, frames)``), so the shorter axis is treated as channels —
    audio frames always vastly outnumber channels in practice.

    Args:
        waveform: A 1-D or 2-D array-like of PCM samples.

    Returns:
        A contiguous 1-D ``float32`` array.

    Raises:
        DataError: if the input has more than two dimensions.
    """
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        return np.ascontiguousarray(array)
    if array.ndim == 2:
        channel_axis = int(np.argmin(array.shape))
        return np.ascontiguousarray(array.mean(axis=channel_axis).astype(np.float32))
    raise DataError(f"expected a 1-D or 2-D waveform, got shape {array.shape}")


def resample_waveform(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono waveform with a polyphase filter (scipy, no librosa needed).

    Args:
        waveform: 1-D mono waveform.
        orig_sr: Original sample rate in Hz.
        target_sr: Desired sample rate in Hz.

    Returns:
        The waveform at ``target_sr`` as ``float32``.

    Raises:
        DataError: if either rate is not a positive integer.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise DataError(f"sample rates must be positive, got {orig_sr} -> {target_sr}")
    array = np.asarray(waveform, dtype=np.float32)
    if orig_sr == target_sr:
        return array
    divisor = math.gcd(int(orig_sr), int(target_sr))
    resampled = resample_poly(array, int(target_sr) // divisor, int(orig_sr) // divisor)
    return np.asarray(resampled, dtype=np.float32)


def normalise_loader_output(result: Any, *, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Normalise a shared-loader return value to mono float32 at ``target_sr``.

    The shared loader contract ("decode + resample to 16 kHz mono") leaves the
    return shape open, so both a bare waveform (assumed already at
    ``target_sr``) and a ``(waveform, sample_rate)`` pair are accepted.

    Args:
        result: Whatever the shared loader returned.
        target_sr: Sample rate the caller requires.

    Returns:
        A 1-D ``float32`` waveform at ``target_sr``.
    """
    if isinstance(result, tuple) and len(result) == 2:
        waveform, rate = result
        return resample_waveform(ensure_mono(waveform), int(rate), target_sr)
    return ensure_mono(result)


def _shared_loader() -> Any | None:
    """Return the decode function from ``tulip.features.audio.loading``, if present."""
    try:
        module = importlib.import_module("tulip.features.audio.loading")
    except Exception:  # module absent or broken: fall back to soundfile
        logger.debug("shared audio loader unavailable; falling back to soundfile")
        return None
    for name in _SHARED_LOADER_CANDIDATES:
        function = getattr(module, name, None)
        if callable(function):
            return function
    logger.debug("tulip.features.audio.loading exposes no known loader function")
    return None


def load_waveform(path: str | Path, *, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode one audio file to a mono float32 waveform at ``sample_rate``.

    Prefers the shared decoder in ``tulip.features.audio.loading`` so speech
    models and audio feature extractors treat files identically; falls back to
    soundfile + scipy polyphase resampling when the shared loader is absent.

    Args:
        path: Audio file to decode.
        sample_rate: Target sample rate in Hz.

    Returns:
        A 1-D ``float32`` waveform.

    Raises:
        MissingDependencyError: if no decoder is available (install ``audio``).
    """
    loader = _shared_loader()
    if loader is not None:
        try:
            result = loader(Path(path), sample_rate=sample_rate)
        except TypeError:  # loader does not take a sample_rate keyword
            result = loader(Path(path))
        return normalise_loader_output(result, target_sr=sample_rate)
    soundfile = optional_import("soundfile", extra="audio", purpose="decoding audio files")
    data, orig_sr = soundfile.read(str(path), dtype="float32", always_2d=False)
    return resample_waveform(ensure_mono(data), int(orig_sr), sample_rate)


def is_whisper_extractor(feature_extractor: Any) -> bool:
    """Whether the extractor produces Whisper's fixed-window ``input_features``.

    Whisper's feature extractor pads/truncates every clip to a fixed 30 s
    log-mel window, so batches must not use dynamic ``padding=True``.
    Detection is by class name to avoid importing transformers here.

    Args:
        feature_extractor: A Hugging Face feature extractor instance.

    Returns:
        ``True`` for Whisper-style extractors.
    """
    return type(feature_extractor).__name__.lower().startswith("whisper")


def _load_clipped_waveforms(
    paths: Sequence[str | Path], *, sample_rate: int, max_seconds: float
) -> list[np.ndarray]:
    """Decode audio files to mono waveforms, clipped to ``max_seconds``.

    Args:
        paths: Audio files to decode.
        sample_rate: Target sample rate in Hz.
        max_seconds: Maximum clip duration retained per file.

    Returns:
        One 1-D ``float32`` waveform per input path.

    Raises:
        DataError: if a file cannot be decoded or decodes to an empty waveform.
    """
    limit = max(1, round(max_seconds * sample_rate))
    waveforms: list[np.ndarray] = []
    for path in paths:
        try:
            waveform = load_waveform(path, sample_rate=sample_rate)
        except TulipError:
            raise
        except Exception as exc:
            raise DataError(f"failed to decode audio file {path}: {exc}") from exc
        if waveform.size == 0:
            raise DataError(f"audio file {path} decoded to an empty waveform")
        waveforms.append(waveform[:limit])
    return waveforms


class FinetunedSpeechClassifier(ClassifierMixin, BaseEstimator):
    """Fine-tune a Hugging Face audio-classification model on audio files.

    Follows scikit-learn conventions: ``fit(paths, y)``, ``predict``,
    ``predict_proba`` (softmax over logits, batched, under ``no_grad``), and a
    ``classes_`` attribute after fitting. Waveforms are decoded per batch
    (never all at once), keeping memory bounded on large corpora at the cost
    of re-decoding each epoch — decoding is cheap next to a transformer
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
        if self.epochs < 1:
            raise ConfigurationError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ConfigurationError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ConfigurationError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ConfigurationError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if self.gradient_accumulation_steps < 1:
            raise ConfigurationError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.class_weight not in (None, "balanced"):
            raise ConfigurationError(
                f'class_weight must be None or "balanced", got {self.class_weight!r}'
            )

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
        if not paths:
            raise DataError("cannot fit on an empty dataset")
        if len(paths) != len(y):
            raise DataError(f"X and y length mismatch: {len(paths)} != {len(y)}")
        classes, encoded = encode_labels(y)
        if len(classes) < 2:
            raise DataError(f"need at least 2 classes to fit, got {len(classes)}")

        device = resolve_device(self.device, torch)
        # Seed BEFORE model construction: from_pretrained randomly initialises
        # the new classification head from the ambient RNG, so seeding only
        # inside the training loop would leave fit() non-reproducible.
        torch.manual_seed(self.seed)
        feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(self.checkpoint)
        id2label = {index: str(label) for index, label in enumerate(classes)}
        model_cls = (
            transformers.WhisperForAudioClassification
            if is_whisper_extractor(feature_extractor)
            else transformers.AutoModelForAudioClassification
        )
        model = model_cls.from_pretrained(
            self.checkpoint,
            num_labels=len(classes),
            id2label=id2label,
            label2id={label: index for index, label in id2label.items()},
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
            waveforms = _load_clipped_waveforms(
                batch_paths, sample_rate=self.sample_rate, max_seconds=self.max_seconds
            )
            features = self._encode_features(feature_extractor, waveforms)
            inputs = {key: value.to(device) for key, value in features.items()}
            targets = torch.as_tensor(encoded[indices], dtype=torch.long, device=device)
            return inputs, targets

        class_weights = None
        if self.class_weight == "balanced":
            class_weights = balanced_class_weights(encoded, len(classes))
        train_torch_classifier(
            torch,
            model,
            encode_batch,
            len(paths),
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
        paths = [Path(path) for path in X]
        if not paths:
            return np.zeros((0, len(self.classes_)), dtype=np.float64)
        self.model_.eval()
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                batch_paths = paths[start : start + self.batch_size]
                waveforms = _load_clipped_waveforms(
                    batch_paths, sample_rate=self.sample_rate, max_seconds=self.max_seconds
                )
                features = self._encode_features(self.feature_extractor_, waveforms)
                inputs = {key: value.to(self.device_) for key, value in features.items()}
                logits = self.model_(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)
                rows.append(probabilities.detach().cpu().numpy())
        return np.vstack(rows).astype(np.float64)

    def predict(self, X: Sequence[str | Path]) -> np.ndarray:
        """Return the most probable class label for each audio file."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


#: Alias matching the architecture document's family name for these wrappers.
SpeechClassifier = FinetunedSpeechClassifier


class EmbeddingSpeechClassifier(ClassifierMixin, BaseEstimator):
    """Frozen speechbrain speaker embeddings + a logistic-regression head.

    This is embedding + head, **not** end-to-end fine-tuning: the pretrained
    encoder (ECAPA-TDNN or x-vector, trained for speaker recognition) is used
    as a fixed feature extractor and only a scikit-learn
    ``LogisticRegression`` head is trained on the pooled utterance embeddings.
    Speaker embeddings carry accent/dialect information while training remains
    cheap enough for CPU-only setups and small corpora.

    Attributes:
        classes_: Sorted array of class labels (after ``fit``).
        embedder_: The frozen speechbrain encoder (after ``fit``).
        head_: The fitted ``LogisticRegression`` head (after ``fit``).
        device_: Device the encoder runs on (after ``fit``).
    """

    def __init__(
        self,
        checkpoint: str = "speechbrain/spkrec-ecapa-voxceleb",
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
        max_seconds: float = 30.0,
        batch_size: int = 8,
        head_c: float = 1.0,
        head_max_iter: int = 1000,
        class_weight: str | None = None,
        savedir: str | Path | None = None,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        """Configure the wrapper; no heavy dependency is imported here.

        Args:
            checkpoint: speechbrain Hub source for the pretrained encoder.
            sample_rate: Encoder input sample rate in Hz.
            max_seconds: Clips are truncated to this many seconds.
            batch_size: Files embedded per encoder forward pass.
            head_c: Inverse regularisation strength of the logistic head.
            head_max_iter: Solver iteration cap for the logistic head.
            class_weight: ``"balanced"`` or ``None``, forwarded to the head.
            savedir: Where speechbrain caches the downloaded encoder; defaults
                to a per-checkpoint directory under the user cache (speechbrain
                would otherwise write into the current working directory).
            device: Explicit device, or ``None`` to use CUDA when available.
            seed: Seed for the logistic-regression solver.

        Note:
            Per the scikit-learn estimator contract, ``__init__`` only stores
            parameters; validation happens in :meth:`fit` so values injected
            via ``set_params`` (e.g. by ``GridSearchCV``) are validated too.
        """
        self.checkpoint = checkpoint
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.batch_size = batch_size
        self.head_c = head_c
        self.head_max_iter = head_max_iter
        self.class_weight = class_weight
        self.savedir = savedir
        self.device = device
        self.seed = seed

    def _validate_hyperparameters(self) -> None:
        """Validate constructor/set_params values (called from :meth:`fit`)."""
        if self.sample_rate < 1:
            raise ConfigurationError(f"sample_rate must be >= 1, got {self.sample_rate}")
        if self.max_seconds <= 0:
            raise ConfigurationError(f"max_seconds must be > 0, got {self.max_seconds}")
        if self.batch_size < 1:
            raise ConfigurationError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.head_c <= 0:
            raise ConfigurationError(f"head_c must be > 0, got {self.head_c}")
        if self.head_max_iter < 1:
            raise ConfigurationError(f"head_max_iter must be >= 1, got {self.head_max_iter}")
        if self.class_weight not in (None, "balanced"):
            raise ConfigurationError(
                f'class_weight must be None or "balanced", got {self.class_weight!r}'
            )

    def _load_encoder(self, torch: Any) -> tuple[Any, str]:
        """Download/load the pretrained speechbrain encoder onto the device."""
        optional_import(
            "speechbrain", extra="speech", purpose="pretrained speaker embedding encoders"
        )
        # EncoderClassifier moved from speechbrain.pretrained to
        # speechbrain.inference in speechbrain 1.0; support both.
        try:
            inference = importlib.import_module("speechbrain.inference")
        except ImportError:
            inference = importlib.import_module("speechbrain.pretrained")
        if self.savedir is not None:
            savedir = Path(self.savedir)
        else:
            cache_name = self.checkpoint.replace("/", "--")
            savedir = Path.home() / ".cache" / "tulip" / "speechbrain" / cache_name
        device = resolve_device(self.device, torch)
        encoder = inference.EncoderClassifier.from_hparams(
            source=self.checkpoint, savedir=str(savedir), run_opts={"device": device}
        )
        return encoder, device

    def _embed(self, torch: Any, encoder: Any, paths: list[Path], device: str) -> np.ndarray:
        """Embed audio files in batches with the frozen encoder."""
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                batch_paths = paths[start : start + self.batch_size]
                waveforms = _load_clipped_waveforms(
                    batch_paths, sample_rate=self.sample_rate, max_seconds=self.max_seconds
                )
                lengths = [waveform.size for waveform in waveforms]
                max_length = max(lengths)
                batch = torch.zeros(len(waveforms), max_length, dtype=torch.float32)
                for row, waveform in enumerate(waveforms):
                    batch[row, : waveform.size] = torch.from_numpy(waveform)
                relative = torch.as_tensor(
                    [length / max_length for length in lengths], dtype=torch.float32
                )
                embeddings = encoder.encode_batch(batch.to(device), relative.to(device))
                array = embeddings.detach().cpu().numpy()
                rows.append(array.reshape(array.shape[0], -1))
        return np.vstack(rows).astype(np.float64)

    def fit(self, X: Sequence[str | Path], y: Sequence[Any]) -> EmbeddingSpeechClassifier:
        """Embed the audio files and train the logistic-regression head.

        Args:
            X: Sequence of audio file paths.
            y: Parallel sequence of labels (any hashables; coerced to str).

        Returns:
            ``self``, fitted.

        Raises:
            ConfigurationError: if a hyperparameter is out of range.
            MissingDependencyError: if torch/speechbrain are not installed.
            DataError: if inputs are empty, mismatched, single-class, or
                undecodable.
        """
        self._validate_hyperparameters()  # before imports: valid config first
        torch = optional_import(
            "torch", extra="speech", purpose="pretrained speaker embedding encoders"
        )
        paths = [Path(path) for path in X]
        if not paths:
            raise DataError("cannot fit on an empty dataset")
        if len(paths) != len(y):
            raise DataError(f"X and y length mismatch: {len(paths)} != {len(y)}")
        labels = np.asarray([str(value) for value in y], dtype=object)
        if len(np.unique(labels)) < 2:
            raise DataError(f"need at least 2 classes to fit, got {len(np.unique(labels))}")

        encoder, device = self._load_encoder(torch)
        logger.info(
            "embedding %d audio files with frozen %s (device=%s)",
            len(paths),
            self.checkpoint,
            device,
        )
        embeddings = self._embed(torch, encoder, paths, device)

        from sklearn.linear_model import LogisticRegression

        head = LogisticRegression(
            C=self.head_c,
            max_iter=self.head_max_iter,
            class_weight=self.class_weight,
            random_state=self.seed,
        )
        head.fit(embeddings, labels)
        self.classes_ = head.classes_
        self.embedder_ = encoder
        self.head_ = head
        self.device_ = device
        return self

    def predict_proba(self, X: Sequence[str | Path]) -> np.ndarray:
        """Return class probabilities from the logistic head over embeddings.

        Args:
            X: Sequence of audio file paths.

        Returns:
            Array of shape ``(len(X), n_classes)``; columns follow ``classes_``.

        Raises:
            TulipError: if the model has not been fitted.
            MissingDependencyError: if torch is not installed.
        """
        require_fitted(self, "embedder_", "head_")
        torch = optional_import("torch", extra="speech", purpose="speaker embedding inference")
        paths = [Path(path) for path in X]
        if not paths:
            return np.zeros((0, len(self.classes_)), dtype=np.float64)
        embeddings = self._embed(torch, self.embedder_, paths, self.device_)
        return np.asarray(self.head_.predict_proba(embeddings), dtype=np.float64)

    def predict(self, X: Sequence[str | Path]) -> np.ndarray:
        """Return the most probable class label for each audio file."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def _register_factories() -> None:
    for name, checkpoint in FINETUNE_CHECKPOINTS.items():
        MODELS.add(
            name,
            checkpoint_factory(FinetunedSpeechClassifier, checkpoint),
            # training_aware: accepts the shared TrainingConfig knobs; the
            # embedding models below do not (frozen encoder + sklearn head).
            metadata={"training_aware": True},
        )
    for name, checkpoint in EMBEDDING_CHECKPOINTS.items():
        MODELS.add(name, checkpoint_factory(EmbeddingSpeechClassifier, checkpoint))


_register_factories()
