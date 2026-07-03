"""Tests for :mod:`tulip.models.neural_audio` that need no optional dependency.

torch/transformers/speechbrain/soundfile are exercised only through failure
paths (``importlib.import_module`` monkeypatched so they appear uninstalled)
or replaced by stubs; the waveform helpers are pure numpy/scipy.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest

from tulip.core.exceptions import (
    ConfigurationError,
    DataError,
    MissingDependencyError,
    TulipError,
)
from tulip.models import MODELS
from tulip.models.neural_audio import (
    EmbeddingSpeechClassifier,
    FinetunedSpeechClassifier,
    SpeechClassifier,
    ensure_mono,
    is_whisper_extractor,
    load_waveform,
    normalise_loader_output,
    resample_waveform,
)

SHARED_LOADER_MODULE = "tulip.features.audio.loading"


def block_imports(monkeypatch: pytest.MonkeyPatch, *blocked: str) -> None:
    """Make ``importlib.import_module`` fail for the given module trees."""
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if any(name == root or name.startswith(root + ".") for root in blocked):
            raise ImportError(f"blocked for test: {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


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
    assert 'pip install "tulip[speech]"' in str(excinfo.value)


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


# --- pure waveform helpers ------------------------------------------------------


def test_ensure_mono_passes_through_1d() -> None:
    waveform = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    mono = ensure_mono(waveform)
    assert mono.ndim == 1
    assert mono.dtype == np.float32
    assert mono == pytest.approx(waveform, abs=1e-6)


def test_ensure_mono_averages_frames_by_channels_layout() -> None:
    # soundfile layout: (frames, channels)
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1)
    assert ensure_mono(stereo) == pytest.approx(np.full(100, 0.5))


def test_ensure_mono_averages_channels_by_frames_layout() -> None:
    # torchaudio layout: (channels, frames)
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=0)
    assert ensure_mono(stereo) == pytest.approx(np.full(100, 0.5))


def test_ensure_mono_rejects_3d_input() -> None:
    with pytest.raises(DataError):
        ensure_mono(np.zeros((2, 3, 4)))


def test_resample_waveform_identity_at_same_rate() -> None:
    waveform = np.linspace(-1, 1, 50, dtype=np.float32)
    assert resample_waveform(waveform, 16_000, 16_000) is not None
    assert resample_waveform(waveform, 16_000, 16_000) == pytest.approx(waveform)


def test_resample_waveform_doubles_length_8k_to_16k() -> None:
    waveform = np.ones(8_000, dtype=np.float32)
    resampled = resample_waveform(waveform, 8_000, 16_000)
    assert resampled.shape == (16_000,)
    assert resampled.dtype == np.float32
    # interior of a constant signal must stay ~constant after polyphase filtering
    assert resampled[100:-100] == pytest.approx(np.ones(15_800), abs=1e-3)


def test_resample_waveform_rejects_nonpositive_rates() -> None:
    with pytest.raises(DataError):
        resample_waveform(np.zeros(10, dtype=np.float32), 0, 16_000)


def test_normalise_loader_output_handles_pair_and_bare_array() -> None:
    pair = (np.ones(100, dtype=np.float32), 8_000)
    assert normalise_loader_output(pair, target_sr=16_000).shape == (200,)
    bare = np.ones(100, dtype=np.float32)
    assert normalise_loader_output(bare, target_sr=16_000).shape == (100,)


def test_is_whisper_extractor_by_class_name() -> None:
    class WhisperFeatureExtractor:
        pass

    class Wav2Vec2FeatureExtractor:
        pass

    assert is_whisper_extractor(WhisperFeatureExtractor()) is True
    assert is_whisper_extractor(Wav2Vec2FeatureExtractor()) is False


# --- decoding entry point --------------------------------------------------------


def _install_fake_shared_loader(monkeypatch: pytest.MonkeyPatch, loader) -> None:
    module = ModuleType(SHARED_LOADER_MODULE)
    module.load_audio = loader
    monkeypatch.setitem(sys.modules, SHARED_LOADER_MODULE, module)


def test_load_waveform_prefers_shared_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_loader(path, *, sample_rate):
        seen["path"] = path
        seen["sample_rate"] = sample_rate
        return np.ones(100, dtype=np.float32), 8_000

    _install_fake_shared_loader(monkeypatch, fake_loader)
    waveform = load_waveform("clip.wav", sample_rate=16_000)
    assert seen["sample_rate"] == 16_000
    assert str(seen["path"]) == "clip.wav"
    assert waveform.shape == (200,)  # 8 kHz pair resampled up to 16 kHz


def test_load_waveform_retries_shared_loader_positionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def positional_only_loader(path):
        return np.zeros(10, dtype=np.float32)

    _install_fake_shared_loader(monkeypatch, positional_only_loader)
    assert load_waveform("clip.wav").shape == (10,)


def test_load_waveform_fallback_requires_soundfile(monkeypatch: pytest.MonkeyPatch) -> None:
    block_imports(monkeypatch, SHARED_LOADER_MODULE, "soundfile")
    with pytest.raises(MissingDependencyError) as excinfo:
        load_waveform("clip.wav")
    assert excinfo.value.extra == "audio"
