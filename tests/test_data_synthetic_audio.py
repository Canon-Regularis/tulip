"""Tests for the generated synthetic *audio* reference corpus.

This corpus is the audio path's zero-acquisition escape hatch, so these tests
guard the two properties that make it worth shipping: it is *reproducible* (a
seed fixes the WAV bytes and the manifest byte for byte) and it carries *real,
learnable* acoustic signal rather than noise. The learnability guard is the
load-bearing one -- a synthetic audio corpus that trains to chance would make
every downstream audio benchmark meaningless. It runs here because the classical
audio features (mfcc/pitch/spectral_centroid) need only librosa/soundfile, which
are installed, not torch/speechbrain.
"""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tulip.config.schemas import SplitConfig
from tulip.core.exceptions import ConfigurationError
from tulip.core.types import TaskType
from tulip.data.loaders.synthetic_audio import SyntheticAudioLoader
from tulip.data.registry import DATASETS
from tulip.data.splitting import speaker_disjoint_split
from tulip.data.synthetic_audio import (
    AUDIO_SUBDIR,
    GENERATOR,
    SOURCE,
    AudioSyntheticSpec,
    generate_audio_corpus,
    write_synthetic_audio_manifest,
)
from tulip.labels.taxonomy import RegionalDialect, family_for

if TYPE_CHECKING:
    from pathlib import Path

#: Small, fast spec for structural assertions (16 clips).
SMALL = AudioSyntheticSpec(n_speakers_per_dialect=2, samples_per_speaker=2, seed=7)


def _wav_bytes(directory: Path) -> dict[str, bytes]:
    """Map each WAV's file name to its raw bytes under ``directory/audio``."""
    return {
        path.name: path.read_bytes() for path in sorted((directory / AUDIO_SUBDIR).glob("*.wav"))
    }


# ------------------------------------------------------------- determinism


def test_same_spec_and_root_yield_identical_wavs_and_manifest(tmp_path: Path) -> None:
    """The published guarantee: same spec + root -> byte-identical output."""
    first_root, second_root = tmp_path / "a", tmp_path / "b"
    manifest_a = write_synthetic_audio_manifest(SMALL, first_root)
    manifest_b = write_synthetic_audio_manifest(SMALL, second_root)

    assert manifest_a.read_bytes() == manifest_b.read_bytes()
    wavs_a, wavs_b = _wav_bytes(first_root), _wav_bytes(second_root)
    assert wavs_a.keys() == wavs_b.keys()
    assert all(wavs_a[name] == wavs_b[name] for name in wavs_a)


def test_different_seed_changes_wavs_and_manifest(tmp_path: Path) -> None:
    other = AudioSyntheticSpec(n_speakers_per_dialect=2, samples_per_speaker=2, seed=8)
    manifest_a = write_synthetic_audio_manifest(SMALL, tmp_path / "a")
    manifest_b = write_synthetic_audio_manifest(other, tmp_path / "b")

    assert manifest_a.read_bytes() != manifest_b.read_bytes()
    wavs_a, wavs_b = _wav_bytes(tmp_path / "a"), _wav_bytes(tmp_path / "b")
    assert wavs_a.keys() == wavs_b.keys()  # same layout, different bytes
    assert any(wavs_a[name] != wavs_b[name] for name in wavs_a)


# ------------------------------------------------------------------ labels


def test_dialect_labels_are_taxonomy_values_and_families_derive(tmp_path: Path) -> None:
    dialect_values = {d.value for d in RegionalDialect}
    for sample in generate_audio_corpus(SMALL, tmp_path):
        assert sample.labels.dialect in dialect_values
        # DialectLabels auto-derives family; it must agree with the taxonomy.
        assert sample.labels.family == family_for(sample.labels.dialect).value


def test_samples_are_audio_only_and_marked_as_generated(tmp_path: Path) -> None:
    for sample in generate_audio_corpus(SMALL, tmp_path):
        assert sample.text is None
        assert sample.audio_path is not None
        assert sample.source == SOURCE
        assert sample.metadata["generator"] == GENERATOR
        assert sample.metadata["spec_seed"] == SMALL.seed


def test_every_class_has_at_least_two_speakers(tmp_path: Path) -> None:
    by_class: dict[str, set[str]] = {}
    for sample in generate_audio_corpus(SMALL, tmp_path):
        by_class.setdefault(sample.labels.dialect, set()).add(sample.speaker_id)
    assert by_class, "corpus must not be empty"
    assert all(len(speakers) >= 2 for speakers in by_class.values())


# --------------------------------------------------------------- real WAVs


def test_clips_are_real_16k_mono_wavs_on_disk(tmp_path: Path) -> None:
    """Every sample points at a decodable 16 kHz mono int16 PCM file."""
    for sample in generate_audio_corpus(SMALL, tmp_path):
        path = sample.audio_path
        assert path is not None and path.is_file()
        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == SMALL.sample_rate
            assert handle.getsampwidth() == 2
            assert handle.getnframes() == round(SMALL.duration_s * SMALL.sample_rate)


def test_clips_decode_through_load_audio(tmp_path: Path) -> None:
    """The features decode via load_audio; a generated clip must round-trip it."""
    pytest.importorskip("soundfile")
    from tulip.features.audio.loading import load_audio

    sample = generate_audio_corpus(SMALL, tmp_path)[0]
    signal = load_audio(sample.audio_path, sample_rate=SMALL.sample_rate)
    assert signal.ndim == 1
    assert signal.dtype == np.float32
    assert signal.shape == (round(SMALL.duration_s * SMALL.sample_rate),)
    assert np.isfinite(signal).all()


# ------------------------------------------------------------- split safety


def test_speaker_disjoint_split_succeeds_and_leaks_no_speaker(tmp_path: Path) -> None:
    corpus = generate_audio_corpus(
        AudioSyntheticSpec(n_speakers_per_dialect=8, samples_per_speaker=4, seed=7), tmp_path
    )
    splits = speaker_disjoint_split(corpus, SplitConfig(seed=42))
    groups = [{s.speaker_id for s in part} for part in splits.as_dict().values()]
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            assert not (left & right)


# ------------------------------------------------------- signal, not noise


@pytest.mark.slow
def test_corpus_is_learnable_well_above_chance(tmp_path: Path) -> None:
    """The guard that makes the whole corpus worth shipping.

    Trains on a speaker-disjoint split, so the score cannot come from speaker
    re-identification -- only the shared class fingerprint (F0/formants/tilt)
    generalises. Chance is 1/4; a healthy corpus lands far above it.
    """
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    from tulip.pipeline.classifier import DialectClassifier

    corpus = generate_audio_corpus(
        AudioSyntheticSpec(n_speakers_per_dialect=8, samples_per_speaker=6, seed=7), tmp_path
    )
    splits = speaker_disjoint_split(corpus, SplitConfig(seed=42))
    classifier = DialectClassifier(
        task=TaskType.AUDIO,
        model="logistic_regression",
        features=["mfcc", "pitch", "spectral_centroid"],
    ).fit(splits.train)

    predictions = classifier.predict_samples(splits.test)
    gold = [s.labels.at_level(classifier.target) for s in splits.test]
    accuracy = sum(p.label == g for p, g in zip(predictions, gold, strict=True)) / len(gold)

    assert accuracy > 0.6, f"synthetic audio corpus lost its signal: accuracy={accuracy:.3f}"


# ------------------------------------------------------------------ loader


def test_loader_is_registered_and_generates_on_demand(tmp_path: Path) -> None:
    assert "synthetic_audio" in DATASETS.names()
    loader = DATASETS.create("synthetic_audio", n_speakers_per_dialect=2, samples_per_speaker=2)
    samples = list(loader.load(tmp_path))  # tmp_path holds no manifest -> generate
    assert samples and loader.is_available(tmp_path)
    assert all(s.audio_path is not None and s.audio_path.is_file() for s in samples)


def test_written_manifest_round_trips_through_the_loader(tmp_path: Path) -> None:
    spec = AudioSyntheticSpec(n_speakers_per_dialect=2, samples_per_speaker=3, seed=7)
    path = write_synthetic_audio_manifest(spec, tmp_path)
    assert path.is_file()

    reread = list(SyntheticAudioLoader(manifest="manifest.jsonl").load(tmp_path))
    original = generate_audio_corpus(spec, tmp_path)
    assert [s.id for s in reread] == [s.id for s in original]
    assert [s.labels.dialect for s in reread] == [s.labels.dialect for s in original]
    # Relative manifest paths must resolve back to the on-disk clips.
    assert all(s.audio_path is not None and s.audio_path.is_file() for s in reread)


def test_loader_raises_when_a_configured_manifest_is_absent(tmp_path: Path) -> None:
    from tulip.core.exceptions import DataError

    with pytest.raises(DataError, match="manifest not found"):
        list(SyntheticAudioLoader(manifest="missing.jsonl").load(tmp_path))


# -------------------------------------------------------------- spec guards


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_speakers_per_dialect": 1},
        {"samples_per_speaker": 0},
        {"duration_s": 0.0},
        {"sample_rate": 0},
        {"jitter": 0.5},
        {"jitter": -0.1},
        {"dialects": ()},
    ],
)
def test_spec_rejects_out_of_range_knobs(kwargs: dict) -> None:
    with pytest.raises(ConfigurationError):
        AudioSyntheticSpec(**kwargs)


def test_unknown_dialect_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="unknown synthetic audio dialect"):
        generate_audio_corpus(AudioSyntheticSpec(dialects=("atlantis",)), tmp_path)
