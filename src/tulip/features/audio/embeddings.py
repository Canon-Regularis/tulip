"""Self-supervised speech embeddings pooled into fixed-size vectors.

Registers ``wav2vec2_embeddings`` in :data:`tulip.features.AUDIO_FEATURES`:
time-averaged hidden states of a Hugging Face wav2vec2-family checkpoint.
torch and transformers belong to the ``speech`` extra and are imported only
inside ``transform``, so constructing (and registering) the extractor never
requires them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from tulip.core.exceptions import TulipError
from tulip.features.audio.loading import DEFAULT_SAMPLE_RATE, load_audio
from tulip.features.registries import AUDIO_FEATURES
from tulip.utils import optional
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = get_logger(__name__)

#: Multilingual wav2vec2 checkpoint covering Polish speech.
DEFAULT_CHECKPOINT = "facebook/wav2vec2-xls-r-300m"

#: Signals shorter than this many samples are zero-padded so the wav2vec2
#: convolutional encoder always produces at least one frame.
_MIN_SAMPLES = 640

__all__ = ["DEFAULT_CHECKPOINT", "Wav2Vec2EmbeddingExtractor"]


@AUDIO_FEATURES.register("wav2vec2_embeddings", metadata={"extra": "speech"})
class Wav2Vec2EmbeddingExtractor(TransformerMixin, BaseEstimator):
    """Mean-pooled wav2vec2 hidden states, one embedding row per audio file.

    Files are processed in padded batches under ``torch.no_grad``; padding
    frames are excluded from the mean using the model's own downsampled
    lengths. The model is loaded lazily on first ``transform`` and cached on
    the instance for reuse across calls.

    Args:
        checkpoint: Hugging Face model id or local path of a
            wav2vec2-family encoder.
        sample_rate: Sample rate expected by the checkpoint.
        batch_size: Number of files embedded per forward pass.
        device: torch device string (``"cpu"``, ``"cuda"``, ...); ``None``
            selects CUDA when available.
        max_seconds: If set, audio is truncated to this length before
            embedding to bound memory use.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        batch_size: int = 4,
        device: str | None = None,
        max_seconds: float | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.device = device
        self.max_seconds = max_seconds

    def fit(self, X: Sequence[str | Path], y: Any = None) -> Wav2Vec2EmbeddingExtractor:
        """Resolve the embedding dimensionality; the encoder stays frozen.

        Only the checkpoint *config* is fetched here (kilobytes, cached), so
        ``get_feature_names_out`` works right after ``fit``, the sklearn
        fitted-state contract every sibling extractor honours, while the
        multi-hundred-MB model weights still load lazily on first
        ``transform``.

        Raises:
            MissingDependencyError: If transformers is not installed
                (install the ``speech`` extra).
        """
        del X, y
        transformers = optional.optional_import(
            "transformers", extra="speech", purpose="wav2vec2 speech embeddings"
        )
        config = transformers.AutoConfig.from_pretrained(self.checkpoint)
        self.hidden_size_ = int(config.hidden_size)
        return self

    def transform(self, X: Sequence[str | Path]) -> np.ndarray:
        """Embed each audio file with the (frozen) wav2vec2 encoder.

        Args:
            X: Sequence of audio file paths.

        Returns:
            Float32 array of shape ``(len(X), hidden_size)``.

        Raises:
            MissingDependencyError: If torch or transformers is not installed
                (install the ``speech`` extra).
            DataError: If a file is missing or cannot be decoded.
        """
        torch = optional.optional_import(
            "torch", extra="speech", purpose="wav2vec2 speech embeddings"
        )
        model, processor, device = self._ensure_model()
        paths = list(X)
        if not paths:
            return np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)
        rows: list[np.ndarray] = []
        for start in range(0, len(paths), max(1, int(self.batch_size))):
            batch = paths[start : start + max(1, int(self.batch_size))]
            signals = [self._load_signal(path) for path in batch]
            inputs = processor(
                signals, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state
            pooled = self._masked_mean(torch, model, hidden, inputs.get("attention_mask"))
            rows.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(rows)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return embedding dimension names (requires the model to be loaded)."""
        del input_features
        hidden_size = getattr(self, "hidden_size_", None)
        if hidden_size is None:
            raise TulipError(
                "embedding dimensionality is unknown until the checkpoint config is "
                "loaded; call fit() first"
            )
        return np.asarray([f"wav2vec2_{i}" for i in range(hidden_size)], dtype=object)

    def _load_signal(self, path: str | Path) -> np.ndarray:
        signal = load_audio(path, sample_rate=self.sample_rate)
        if self.max_seconds is not None:
            signal = signal[: max(1, int(self.max_seconds * self.sample_rate))]
        if signal.size < _MIN_SAMPLES:
            signal = np.pad(signal, (0, _MIN_SAMPLES - signal.size))
        return signal

    def _ensure_model(self) -> tuple[Any, Any, Any]:
        """Load (or reuse) the checkpoint; returns ``(model, processor, device)``."""
        torch = optional.optional_import(
            "torch", extra="speech", purpose="wav2vec2 speech embeddings"
        )
        transformers = optional.optional_import(
            "transformers", extra="speech", purpose="wav2vec2 speech embeddings"
        )
        if (
            getattr(self, "_model", None) is None
            or getattr(self, "_loaded_checkpoint", None) != self.checkpoint
        ):
            logger.info("loading wav2vec2 checkpoint %s", self.checkpoint)
            processor = transformers.AutoFeatureExtractor.from_pretrained(self.checkpoint)
            model = transformers.AutoModel.from_pretrained(self.checkpoint)
            model.eval()
            device_name = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(device_name)
            model.to(device)
            self._model = model
            self._processor = processor
            self._torch_device = device
            self._loaded_checkpoint = self.checkpoint
            self.hidden_size_ = int(model.config.hidden_size)
        return self._model, self._processor, self._torch_device

    @staticmethod
    def _masked_mean(torch: Any, model: Any, hidden: Any, attention_mask: Any) -> Any:
        """Mean over time excluding padded frames when lengths are recoverable."""
        get_lengths = getattr(model, "_get_feat_extract_output_lengths", None)
        if attention_mask is None or get_lengths is None:
            return hidden.mean(dim=1)
        lengths = get_lengths(attention_mask.sum(-1)).to(torch.long)
        frame_index = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
        mask = (frame_index < lengths[:, None]).unsqueeze(-1).to(hidden.dtype)
        denominator = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denominator
