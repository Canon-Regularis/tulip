"""Tests for :mod:`tulip.models.neural_audio` that need no optional dependency.

torch/transformers/speechbrain are exercised only through failure paths
(``importlib.import_module`` monkeypatched so they appear uninstalled) or
replaced by stubs; decoding goes through the canonical shared loader, tested
in ``test_features_audio_loading.py``.
"""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING

import numpy as np
import pytest

from conftest import block_imports
from tulip.core.exceptions import ConfigurationError, MissingDependencyError, TulipError
from tulip.models import MODELS
from tulip.models.neural_audio import (
    EmbeddingSpeechClassifier,
    FinetunedSpeechClassifier,
    SpeechClassifier,
    is_whisper_extractor,
)

if TYPE_CHECKING:
    from pathlib import Path

# --- factory / hyperparameter plumbing ----------------------------------------


def test_finetuned_factory_params_land_on_wrapper() -> None:
    model = MODELS.create(
        "wav2vec2",
        max_seconds=6.0,
        epochs=2,
        batch_size=4,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        gradient_accumulation_steps=4,
        class_weight="balanced",
        freeze_feature_encoder=False,
        device="cpu",
        seed=11,
    )
    assert isinstance(model, FinetunedSpeechClassifier)
    assert model.checkpoint == "facebook/wav2vec2-xls-r-300m"
    assert model.max_seconds == pytest.approx(6.0)
    assert model.epochs == 2
    assert model.batch_size == 4
    assert model.learning_rate == pytest.approx(1e-5)
    assert model.warmup_ratio == pytest.approx(0.05)
    assert model.gradient_accumulation_steps == 4
    assert model.class_weight == "balanced"
    assert model.freeze_feature_encoder is False
    assert model.device == "cpu"
    assert model.seed == 11


def test_embedding_factory_params_land_on_wrapper(tmp_path) -> None:
    model = MODELS.create(
        "ecapa_tdnn",
        batch_size=2,
        head_c=0.5,
        head_max_iter=200,
        class_weight="balanced",
        savedir=tmp_path,
        device="cpu",
        seed=3,
    )
    assert isinstance(model, EmbeddingSpeechClassifier)
    assert model.checkpoint == "speechbrain/spkrec-ecapa-voxceleb"
    assert model.batch_size == 2
    assert model.head_c == pytest.approx(0.5)
    assert model.head_max_iter == 200
    assert model.class_weight == "balanced"
    assert model.savedir == tmp_path
    assert model.device == "cpu"
    assert model.seed == 3


def test_speech_factory_checkpoint_is_overridable() -> None:
    assert MODELS.create("hubert", checkpoint="my/hubert").checkpoint == "my/hubert"
    assert MODELS.create("xvector", checkpoint="my/xvector").checkpoint == "my/xvector"


def test_speech_classifier_alias() -> None:
    assert SpeechClassifier is FinetunedSpeechClassifier


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_seconds": 0.0},
        {"sample_rate": 0},
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": -1.0},
        {"warmup_ratio": 2.0},
        {"gradient_accumulation_steps": 0},
        {"class_weight": "focal"},
    ],
)
def test_finetuned_fit_rejects_bad_hyperparameters(kwargs: dict) -> None:
    # Validated in fit (sklearn estimator contract: set_params-injected values
    # must be validated too), before the optional torch import, so this runs
    # on torch-less machines.
    model = FinetunedSpeechClassifier(**kwargs)
    with pytest.raises(ConfigurationError):
        model.fit(["a.wav", "b.wav"], ["podhale", "silesia"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"max_seconds": -1.0},
        {"batch_size": 0},
        {"head_c": 0.0},
        {"head_max_iter": 0},
        {"class_weight": "focal"},
    ],
)
def test_embedding_fit_rejects_bad_hyperparameters(kwargs: dict) -> None:
    model = EmbeddingSpeechClassifier(**kwargs)
    with pytest.raises(ConfigurationError):
        model.fit(["a.wav", "b.wav"], ["podhale", "silesia"])


# --- missing-dependency behaviour ---------------------------------------------


def test_finetuned_fit_without_torch_names_speech_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_imports(monkeypatch, "torch", "transformers")
    model = MODELS.create("wav2vec2")
    with pytest.raises(MissingDependencyError) as excinfo:
        model.fit(["a.wav", "b.wav"], ["podhale", "silesia"])
    assert excinfo.value.extra == "speech"
    assert 'pip install "tulip-dialect[speech]"' in str(excinfo.value)


def test_embedding_fit_without_torch_names_speech_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_imports(monkeypatch, "torch", "speechbrain")
    model = MODELS.create("xvector")
    with pytest.raises(MissingDependencyError) as excinfo:
        model.fit(["a.wav", "b.wav"], ["podhale", "silesia"])
    assert excinfo.value.extra == "speech"


def test_predict_before_fit_raises_tulip_error() -> None:
    with pytest.raises(TulipError, match="not fitted"):
        FinetunedSpeechClassifier().predict(["a.wav"])
    with pytest.raises(TulipError, match="not fitted"):
        EmbeddingSpeechClassifier().predict(["a.wav"])


# --- decoding through the canonical shared loader --------------------------------


def _write_wav(path: Path, *, seconds: float, framerate: int) -> Path:
    samples = np.zeros(int(seconds * framerate), dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(framerate)
        handle.writeframes(samples.tobytes())
    return path


def test_clipped_waveforms_honour_sample_rate_through_real_loader(tmp_path: Path) -> None:
    # Regression: an earlier local decode path silently ignored sample_rate
    # (it called the shared loader with the wrong keyword and fell back to
    # the 16 kHz default). Decoding a 1 s 8 kHz file AT 8 kHz must yield
    # 8000 samples, not 16000.
    pytest.importorskip("soundfile")
    from tulip.models.neural_audio import _load_clipped_waveforms

    clip = _write_wav(tmp_path / "clip.wav", seconds=1.0, framerate=8_000)
    (waveform,) = _load_clipped_waveforms([clip], sample_rate=8_000, max_seconds=5.0)
    assert waveform.shape == (8_000,)

    (upsampled,) = _load_clipped_waveforms([clip], sample_rate=16_000, max_seconds=5.0)
    assert upsampled.shape == (16_000,)

    # max_seconds clips at the requested rate.
    (clipped,) = _load_clipped_waveforms([clip], sample_rate=8_000, max_seconds=0.5)
    assert clipped.shape == (4_000,)


def test_is_whisper_extractor_by_class_name() -> None:
    class WhisperFeatureExtractor:
        pass

    class Wav2Vec2FeatureExtractor:
        pass

    assert is_whisper_extractor(WhisperFeatureExtractor()) is True
    assert is_whisper_extractor(Wav2Vec2FeatureExtractor()) is False
