"""Frozen speaker-embedding speech models (speechbrain encoder + sklearn head).

:class:`EmbeddingSpeechClassifier` extracts frozen pretrained speaker embeddings
(speechbrain ECAPA-TDNN / x-vector) and fits a scikit-learn
``LogisticRegression`` head on top. This is embedding + head, **not** end-to-end
fine-tuning: the encoder weights never change, which is far cheaper than the
fine-tuning wrapper in :mod:`tulip.models.neural_audio_finetune` and works well
on small dialect corpora. The two share :mod:`tulip.models._audio_common`.

It follows scikit-learn conventions (``fit(paths, y)`` / ``predict`` /
``predict_proba`` / ``classes_``) over sequences of audio file paths. All heavy
dependencies (torch, speechbrain, soundfile) are imported lazily inside methods,
so importing this module never requires an optional dependency.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tulip.core.exceptions import ConfigurationError
from tulip.models._audio_common import TARGET_SAMPLE_RATE, load_clipped_waveforms
from tulip.models._common import (
    ArgmaxPredictMixin,
    checkpoint_factory,
    empty_proba,
    require_fitted,
    resolve_device,
    validate_class_weight,
    validate_fit_inputs,
)
from tulip.models.registry import MODELS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

#: Registry name -> speechbrain source for the embedding + head models.
EMBEDDING_CHECKPOINTS: dict[str, str] = {
    "ecapa_tdnn": "speechbrain/spkrec-ecapa-voxceleb",
    "xvector": "speechbrain/spkrec-xvect-voxceleb",
}

__all__ = ["EMBEDDING_CHECKPOINTS", "EmbeddingSpeechClassifier"]


class EmbeddingSpeechClassifier(ArgmaxPredictMixin, ClassifierMixin, BaseEstimator):
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
        validate_class_weight(self)

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
                waveforms = load_clipped_waveforms(
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
        classes, encoded = validate_fit_inputs(paths, y)
        labels = classes[encoded]  # per-sample string labels for the sklearn head

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
            return empty_proba(len(self.classes_))
        embeddings = self._embed(torch, self.embedder_, paths, self.device_)
        return np.asarray(self.head_.predict_proba(embeddings), dtype=np.float64)


def _register_factories() -> None:
    for name, checkpoint in EMBEDDING_CHECKPOINTS.items():
        MODELS.add(
            name,
            checkpoint_factory(EmbeddingSpeechClassifier, checkpoint),
            metadata={"raw_input": True, "extra": "speech"},
        )


_register_factories()
