"""Registry-level tests for the neural text, neural speech, and fastText models.

These must pass with none of the optional dependencies installed: importing
:mod:`tulip.models` registers everything, and instantiating any model via the
registry never touches torch/transformers/speechbrain/fasttext.
"""

from __future__ import annotations

import pytest

from tulip.models import MODELS
from tulip.models.fasttext_model import FastTextClassifier
from tulip.models.neural_audio import (
    EMBEDDING_CHECKPOINTS,
    FINETUNE_CHECKPOINTS,
    EmbeddingSpeechClassifier,
    FinetunedSpeechClassifier,
)
from tulip.models.neural_text import TEXT_CHECKPOINTS, TransformerTextClassifier

NEURAL_MODEL_NAMES = (
    "herbert",
    "polish_roberta",
    "mbert",
    "xlm_roberta",
    "fasttext",
    "wav2vec2",
    "hubert",
    "whisper",
    "ecapa_tdnn",
    "xvector",
)

EXPECTED_TYPES: dict[str, type] = {
    "herbert": TransformerTextClassifier,
    "polish_roberta": TransformerTextClassifier,
    "mbert": TransformerTextClassifier,
    "xlm_roberta": TransformerTextClassifier,
    "fasttext": FastTextClassifier,
    "wav2vec2": FinetunedSpeechClassifier,
    "hubert": FinetunedSpeechClassifier,
    "whisper": FinetunedSpeechClassifier,
    "ecapa_tdnn": EmbeddingSpeechClassifier,
    "xvector": EmbeddingSpeechClassifier,
}


@pytest.mark.parametrize("name", NEURAL_MODEL_NAMES)
def test_registry_contains_all_neural_names(name: str) -> None:
    assert name in MODELS


@pytest.mark.parametrize("name", NEURAL_MODEL_NAMES)
def test_create_requires_no_optional_dependencies(name: str) -> None:
    model = MODELS.create(name)
    assert isinstance(model, EXPECTED_TYPES[name])


def test_text_factories_bind_canonical_checkpoints() -> None:
    assert TEXT_CHECKPOINTS == {
        "herbert": "allegro/herbert-base-cased",
        "polish_roberta": "sdadas/polish-roberta-base-v2",
        "mbert": "bert-base-multilingual-cased",
        "xlm_roberta": "xlm-roberta-base",
    }
    for name, checkpoint in TEXT_CHECKPOINTS.items():
        assert MODELS.create(name).checkpoint == checkpoint


def test_speech_factories_bind_canonical_checkpoints() -> None:
    assert FINETUNE_CHECKPOINTS == {
        "wav2vec2": "facebook/wav2vec2-xls-r-300m",
        "hubert": "facebook/hubert-base-ls960",
        "whisper": "openai/whisper-small",
    }
    assert EMBEDDING_CHECKPOINTS == {
        "ecapa_tdnn": "speechbrain/spkrec-ecapa-voxceleb",
        "xvector": "speechbrain/spkrec-xvect-voxceleb",
    }
    for name, checkpoint in {**FINETUNE_CHECKPOINTS, **EMBEDDING_CHECKPOINTS}.items():
        assert MODELS.create(name).checkpoint == checkpoint


def test_classifiers_expose_sklearn_style_api() -> None:
    for name in NEURAL_MODEL_NAMES:
        model = MODELS.create(name)
        assert callable(model.fit)
        assert callable(model.predict)
        assert callable(model.predict_proba)
