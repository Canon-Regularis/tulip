"""Audio feature extraction: pooled classical descriptors and speech embeddings.

Importing this package registers every built-in audio feature extractor in
:data:`tulip.features.AUDIO_FEATURES` under the canonical names ``mfcc``,
``mel_spectrogram``, ``pitch``, ``formants``, ``energy``,
``zero_crossing_rate``, ``spectral_centroid``, ``chroma``, and
``wav2vec2_embeddings``. All extractors are sklearn transformers over
sequences of audio file paths, returning one dense float32 row per file.
Heavy dependencies (librosa, soundfile, parselmouth, torch, transformers)
are imported lazily inside methods, so importing this package never requires
an optional dependency.
"""

from __future__ import annotations

from tulip.features.audio.composite import build_audio_features
from tulip.features.audio.embeddings import DEFAULT_CHECKPOINT, Wav2Vec2EmbeddingExtractor
from tulip.features.audio.loading import DEFAULT_SAMPLE_RATE, load_audio, resample
from tulip.features.audio.pooling import (
    DEFAULT_STATS,
    VALID_STATS,
    pool_features,
    pooled_feature_names,
)
from tulip.features.audio.prosody import FormantExtractor, PitchExtractor, lpc_formant_frames
from tulip.features.audio.spectral import (
    ChromaExtractor,
    MelSpectrogramExtractor,
    MfccExtractor,
    RmsEnergyExtractor,
    SpectralCentroidExtractor,
    ZeroCrossingRateExtractor,
)

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_STATS",
    "VALID_STATS",
    "ChromaExtractor",
    "FormantExtractor",
    "MelSpectrogramExtractor",
    "MfccExtractor",
    "PitchExtractor",
    "RmsEnergyExtractor",
    "SpectralCentroidExtractor",
    "Wav2Vec2EmbeddingExtractor",
    "ZeroCrossingRateExtractor",
    "build_audio_features",
    "load_audio",
    "lpc_formant_frames",
    "pool_features",
    "pooled_feature_names",
    "resample",
]
