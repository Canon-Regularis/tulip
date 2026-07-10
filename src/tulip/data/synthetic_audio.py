"""Procedural generator for a synthetic *audio* dialect corpus (source-filter).

The toolkit ships nine classical audio features and several speech models, yet
nothing exercises the audio path end to end without acquiring real speech. This
module is the audio analogue of :mod:`tulip.data.synthetic`: a deterministic
generator that synthesises short, learnable, dialect-correlated WAV clips so
``tulip train configs/synthetic_audio.yaml`` runs on a clean checkout with zero
data acquisition.

Each synthetic "dialect" is a distinct point in acoustic space (see
:mod:`tulip.data._synthetic_audio_corpus`): a fundamental-frequency register, a
formant triple, and a spectral-tilt pole. A clip is built with a classic
source-filter (Klatt-style) model:

1. A glottal **source** -- an impulse train at the class F0, phase-accumulated
   through a slow vibrato so the pitch contour is not dead flat.
2. A **filter** -- three parallel second-order bandpass resonators at the class
   formants (F1/F2/F3), summed with decreasing weight, then a one-pole tilt
   filter imposing the class spectral slope.
3. A little seeded aspiration **noise**, and a constant-loudness normalisation
   so loudness never leaks class identity.

Small per-speaker jitter is applied to F0 and the formants, so speakers within a
class differ (which is what makes a speaker-disjoint split meaningful) while the
class fingerprint dominates. The result is separable by the *classical* audio
features (``pitch``, ``formants``, ``spectral_centroid``, ``mfcc``) with a
logistic-regression head -- no torch/speechbrain required.

Generation is fully deterministic: one ``numpy.random.default_rng(seed)`` is
consumed in a fixed order (sorted class keys -> speaker index -> sample index,
and within a sample: vibrato phase, then the noise vector), and clips are
written as int16 PCM with **no dithering**, so the same
:class:`AudioSyntheticSpec` and ``root`` always reproduce byte-identical WAVs
and manifest. WAVs are written with the standard-library :mod:`wave` module, so
generation needs no audio extra; only *reading* the clips back (in the features)
needs soundfile/librosa.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import DialectLabels, Sample
from tulip.data._synthetic_audio_corpus import (
    DIALECT_ACOUSTICS,
    DIALECTS,
    FORMANT_BANDWIDTHS,
    FORMANT_WEIGHTS,
    NOISE_RMS_FRACTION,
    TARGET_RMS,
    VIBRATO_DEPTH,
    VIBRATO_RATE_HZ,
)
from tulip.utils.io import write_jsonl
from tulip.utils.logging import get_logger

__all__ = [
    "AudioSyntheticSpec",
    "generate_audio_corpus",
    "write_synthetic_audio_manifest",
]

_logger = get_logger(__name__)

#: Value recorded in :attr:`Sample.source` for every generated clip.
SOURCE = "synthetic_audio"

#: Value recorded in ``Sample.metadata["generator"]``.
GENERATOR = "tulip-synthetic-audio"

#: Sub-directory of ``root`` the WAV clips are written under. Manifest audio
#: paths are stored relative to ``root`` as ``audio/<id>.wav`` (POSIX
#: separators) so a materialised corpus is relocatable and byte-stable.
AUDIO_SUBDIR = "audio"

_INT16_MAX = 32_767
_INT16_MIN = -32_768


@dataclass(frozen=True)
class AudioSyntheticSpec:
    """Knobs controlling synthetic audio-corpus generation.

    Args:
        n_speakers_per_dialect: Distinct speakers generated per class. Must be
            >= 2 so a speaker-disjoint split has more than one group per class
            (the corpus's whole point).
        samples_per_speaker: Clips generated per speaker.
        dialects: Class keys to include (``None`` selects every class in
            :data:`~tulip.data._synthetic_audio_corpus.DIALECT_ACOUSTICS`). Keys
            are taxonomy ``RegionalDialect`` values (e.g. ``"podhale"``).
        duration_s: Clip duration in seconds.
        sample_rate: Output sample rate in Hz (16 kHz matches the toolkit's
            audio features).
        jitter: Per-speaker relative jitter applied to F0 and the formants
            (e.g. ``0.06`` = +/-6 %). Kept small enough that the class F0
            registers never overlap.
        seed: Seed for the single generator RNG; fixes the entire output.

    Raises:
        ConfigurationError: if any knob is out of range.
    """

    n_speakers_per_dialect: int = 6
    samples_per_speaker: int = 6
    dialects: tuple[str, ...] | None = None
    duration_s: float = 0.8
    sample_rate: int = 16_000
    jitter: float = 0.06
    seed: int = 7

    def __post_init__(self) -> None:
        if self.n_speakers_per_dialect < 2:
            raise ConfigurationError(
                "n_speakers_per_dialect must be >= 2 so speaker-disjoint splitting "
                f"is meaningful, got {self.n_speakers_per_dialect}"
            )
        if self.samples_per_speaker < 1:
            raise ConfigurationError(
                f"samples_per_speaker must be >= 1, got {self.samples_per_speaker}"
            )
        if self.duration_s <= 0.0:
            raise ConfigurationError(f"duration_s must be positive, got {self.duration_s}")
        if self.sample_rate <= 0:
            raise ConfigurationError(f"sample_rate must be positive, got {self.sample_rate}")
        if not 0.0 <= self.jitter < 0.5:
            # A jitter of 0.5 would let adjacent F0 registers (spaced ~30 %)
            # overlap, dissolving the very class signal the corpus exists for.
            raise ConfigurationError(f"jitter must be within [0, 0.5), got {self.jitter}")
        if self.dialects is not None and not self.dialects:
            raise ConfigurationError("dialects must be None or a non-empty sequence of keys")


def generate_audio_corpus(spec: AudioSyntheticSpec, root: Path) -> list[Sample]:
    """Synthesize the corpus described by ``spec``, writing clips under ``root``.

    Clips are written to ``root/audio/<id>.wav`` and the returned
    :class:`Sample` objects carry the absolute WAV path. The output is
    deterministic for a given ``spec`` and ``root``: one RNG is consumed in a
    fixed order (sorted class keys, speaker index, sample index; within a sample
    the vibrato phase then the noise vector), and int16 quantisation is applied
    without dithering, so two calls reproduce byte-identical WAVs.

    Args:
        spec: Generation knobs (see :class:`AudioSyntheticSpec`).
        root: Directory the ``audio/`` sub-directory is created under.

    Returns:
        The generated :class:`Sample` list, classes in sorted order.

    Raises:
        ConfigurationError: if ``spec.dialects`` names an unknown class key.
    """
    root = Path(root)
    audio_dir = root / AUDIO_SUBDIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_dialects(spec.dialects)

    rng = np.random.default_rng(spec.seed)
    samples: list[Sample] = []
    for key in selected:
        f0, formants, tilt, region, voivodeships = DIALECT_ACOUSTICS[key]
        voivodeship = voivodeships[0] if voivodeships else None
        for speaker_index in range(spec.n_speakers_per_dialect):
            speaker_id = f"{SOURCE}-{key}-spk{speaker_index:02d}"
            # Per-speaker jitter is drawn once (fixed order: F0 then the three
            # formants) so every clip from this speaker shares one voice, which
            # is what a speaker-disjoint split must generalise across.
            f0_speaker = f0 * (1.0 + spec.jitter * float(rng.uniform(-1.0, 1.0)))
            formant_scale = 1.0 + spec.jitter * rng.uniform(-1.0, 1.0, size=len(formants))
            formants_speaker = tuple(
                float(freq * scale) for freq, scale in zip(formants, formant_scale, strict=True)
            )
            for sample_index in range(spec.samples_per_speaker):
                clip = _synthesize_clip(
                    rng,
                    f0=f0_speaker,
                    formants=formants_speaker,
                    tilt=tilt,
                    duration_s=spec.duration_s,
                    sample_rate=spec.sample_rate,
                )
                sample_id = f"{SOURCE}-{key}-spk{speaker_index:02d}-{sample_index:03d}"
                path = audio_dir / f"{sample_id}.wav"
                _write_wav(path, clip, spec.sample_rate)
                samples.append(
                    Sample(
                        id=sample_id,
                        text=None,
                        audio_path=path,
                        speaker_id=speaker_id,
                        labels=DialectLabels(dialect=key, region=region, voivodeship=voivodeship),
                        source=SOURCE,
                        metadata={"generator": GENERATOR, "spec_seed": spec.seed},
                    )
                )

    _logger.info(
        "generated %d synthetic audio clips across %d classes under %s (seed=%d)",
        len(samples),
        len(selected),
        audio_dir,
        spec.seed,
    )
    return samples


def write_synthetic_audio_manifest(spec: AudioSyntheticSpec, root: Path) -> Path:
    """Generate the clips and persist a ``root/manifest.jsonl`` describing them.

    The manifest is written in the flat, one-object-per-line shape that
    :func:`tulip.data.manifest.read_manifest` consumes (audio paths relative to
    ``root``), so a generated corpus can be checked in and re-loaded from its
    auditable manifest instead of regenerated.

    Args:
        spec: Generation knobs.
        root: Directory to write the clips and manifest under (created if
            absent).

    Returns:
        The path to the written ``manifest.jsonl``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    corpus = generate_audio_corpus(spec, root)
    path = root / "manifest.jsonl"
    written = write_jsonl(path, (_to_manifest_record(sample, root) for sample in corpus))
    _logger.info("wrote %d synthetic audio samples to %s", written, path)
    return path


def _to_manifest_record(sample: Sample, root: Path) -> dict[str, object]:
    """Flatten one :class:`Sample` into a read_manifest-compatible record.

    The audio path is stored relative to ``root`` with POSIX separators so the
    manifest is byte-identical regardless of the absolute directory or platform.
    """
    audio_path = Path(sample.audio_path) if sample.audio_path is not None else root
    record: dict[str, object] = {
        "id": sample.id,
        "audio_path": audio_path.relative_to(root).as_posix(),
        "speaker_id": sample.speaker_id,
    }
    for field in ("family", "dialect", "region", "village", "voivodeship"):
        value = getattr(sample.labels, field)
        if value is not None:
            record[field] = value
    record["generator"] = sample.metadata["generator"]
    record["spec_seed"] = sample.metadata["spec_seed"]
    return record


def _select_dialects(dialects: tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve the requested class keys against the acoustic table, sorted."""
    if dialects is None:
        return DIALECTS
    chosen = tuple(sorted({key.strip().lower() for key in dialects}))
    unknown = [key for key in chosen if key not in DIALECT_ACOUSTICS]
    if unknown:
        raise ConfigurationError(
            f"unknown synthetic audio dialect key(s): {', '.join(unknown)}; "
            f"available keys: {', '.join(DIALECTS)}"
        )
    return chosen


def _synthesize_clip(
    rng: np.random.Generator,
    *,
    f0: float,
    formants: tuple[float, ...],
    tilt: float,
    duration_s: float,
    sample_rate: int,
) -> np.ndarray:
    """Synthesize one source-filter clip as a constant-loudness float signal.

    RNG draws happen in a fixed order -- the vibrato phase, then the whole noise
    vector -- so the stream stays reproducible regardless of the acoustic
    parameters.
    """
    n_samples = round(duration_s * sample_rate)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    noise = rng.standard_normal(n_samples)

    # Source: an impulse train at a vibrato-modulated F0, placed by wrapping an
    # accumulated instantaneous phase (so period varies smoothly, not abruptly).
    times = np.arange(n_samples) / sample_rate
    inst_f0 = f0 * (1.0 + VIBRATO_DEPTH * np.sin(2.0 * np.pi * VIBRATO_RATE_HZ * times + phase))
    cycles = np.cumsum(inst_f0 / sample_rate)
    whole_cycle = np.floor(cycles)
    is_pulse = np.diff(whole_cycle, prepend=whole_cycle[0] - 1.0) > 0.0
    source = is_pulse.astype(np.float64)

    # Filter: parallel formant resonators, then a one-pole spectral tilt.
    voiced = np.zeros(n_samples, dtype=np.float64)
    for freq, bandwidth, weight in zip(formants, FORMANT_BANDWIDTHS, FORMANT_WEIGHTS, strict=True):
        voiced += weight * _bandpass(source, freq, bandwidth, sample_rate)
    tilted = np.asarray(lfilter([1.0 - tilt], [1.0, -tilt], voiced), dtype=np.float64)

    # Aspiration noise at a fixed fraction of the voiced RMS, then normalise so
    # every clip has the same loudness (loudness must not encode class identity).
    signal = tilted + NOISE_RMS_FRACTION * _rms(tilted) * noise
    signal_rms = _rms(signal)
    if signal_rms > 0.0:
        signal = signal * (TARGET_RMS / signal_rms)
    return signal


def _bandpass(source: np.ndarray, freq: float, bandwidth: float, sample_rate: int) -> np.ndarray:
    """Apply an RBJ constant-peak-gain bandpass biquad (a formant resonator)."""
    w0 = 2.0 * np.pi * freq / sample_rate
    alpha = np.sin(w0) / (2.0 * (freq / bandwidth))
    b = np.array([alpha, 0.0, -alpha])
    a = np.array([1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha])
    return np.asarray(lfilter(b / a[0], a / a[0], source), dtype=np.float64)


def _rms(signal: np.ndarray) -> float:
    """Root-mean-square level of ``signal`` (0.0 for an empty array)."""
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal))))


def _write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    """Quantise ``signal`` to int16 PCM (no dithering) and write a mono WAV.

    The explicit little-endian ``<i2`` dtype makes the bytes platform
    independent, which the determinism guarantee relies on.
    """
    quantised = np.clip(np.round(signal * _INT16_MAX), _INT16_MIN, _INT16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(quantised.tobytes())
