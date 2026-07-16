"""Neural speech classifiers for dialect identification from raw audio (facade).

Two complementary strategies, each now in its own focused module, re-exported
here for backward compatibility:

* :class:`FinetunedSpeechClassifier`
  (:mod:`tulip.models.neural_audio_finetune`) fine-tunes a Hugging Face
  audio-classification model end to end (wav2vec2, HuBERT, Whisper).
* :class:`EmbeddingSpeechClassifier`
  (:mod:`tulip.models.neural_audio_embedding`) fits a logistic-regression head
  on frozen speechbrain speaker embeddings (ECAPA-TDNN, x-vector).

The shared decode and Whisper-detection helpers live in
:mod:`tulip.models._audio_common`. Importing this module registers both
families' factories (wav2vec2, hubert, whisper, ecapa_tdnn, xvector) in
:data:`tulip.models.MODELS`, exactly as before the split.
"""

from __future__ import annotations

from tulip.models._audio_common import (
    TARGET_SAMPLE_RATE,
    is_whisper_extractor,
    load_clipped_waveforms,
)
from tulip.models.neural_audio_embedding import (
    EMBEDDING_CHECKPOINTS,
    EmbeddingSpeechClassifier,
)
from tulip.models.neural_audio_finetune import (
    FINETUNE_CHECKPOINTS,
    FinetunedSpeechClassifier,
    SpeechClassifier,
)

__all__ = [
    "EMBEDDING_CHECKPOINTS",
    "FINETUNE_CHECKPOINTS",
    "TARGET_SAMPLE_RATE",
    "EmbeddingSpeechClassifier",
    "FinetunedSpeechClassifier",
    "SpeechClassifier",
    "is_whisper_extractor",
    "load_clipped_waveforms",
]
