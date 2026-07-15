"""Acoustic resource for the synthetic *audio* dialect generator.

This module is the *data* half of :mod:`tulip.data.synthetic_audio`: the literal
per-class acoustic fingerprints the source-filter synthesiser draws from. Each
synthetic "dialect" is given a distinct point in acoustic space: a fundamental
frequency (pitch register), a formant triple (vowel-space centroid), and a
spectral-tilt pole (how fast the source harmonics roll off), chosen so that
the toolkit's *classical* audio features separate the classes:

* **F0** drives the ``pitch`` feature (f0_mean/std/min/max/range/voiced_ratio).
  The four registers are spaced ~30 % apart so that, even with the per-speaker
  jitter applied at generation time, the classes never overlap in F0 and the
  corpus stays learnable from pitch alone.
* **Formants F1/F2/F3** drive the ``formants`` feature and shape the ``mfcc``
  cepstral envelope and the ``mel_spectrogram``. The classes range from a
  back/low vowel space (high F1, low F2: ``podhale``) to a front/high one
  (low F1, high F2: ``mazovia_proper``).
* **Spectral tilt** (a one-pole lowpass coefficient) drives the
  ``spectral_centroid`` and the low MFCCs: a darker source (pole near 1) pushes
  the spectral centre of mass down, a brighter one lifts it.

Like :mod:`tulip.data._synthetic_corpus`, this module contains **zero logic**
and imports nothing but ``__future__``: no numpy, no scipy, so importing
``tulip.data`` stays light and pulls in none of the scientific stack.

Ordering is load-bearing. The generator consumes a single
``numpy.random.default_rng(seed)`` in a fixed order and indexes into these
tuples by position, so tuple order (and the sorted class-key order derived from
:data:`DIALECT_ACOUSTICS`) is part of the published determinism guarantee and
must not be reordered. In particular the formant triples list F1, F2, F3 in
that order, matching :data:`FORMANT_BANDWIDTHS` and :data:`FORMANT_WEIGHTS`.
"""

from __future__ import annotations

#: Class keys are taxonomy ``RegionalDialect`` values, so
#: ``DialectLabels(dialect=key)`` auto-derives a family. The four chosen classes
#: also fall in four *distinct* families (lesser_polish, silesian, kashubian,
#: masovian), which keeps a family-level view of the corpus balanced too.
#:
#: Each value is
#: ``(f0_hz, (F1, F2, F3) Hz, tilt_pole, region, (voivodeship, ...))`` where
#: ``tilt_pole`` is the coefficient of a one-pole lowpass in ``[0, 1)``: larger
#: is darker (more low-frequency emphasis, lower spectral centroid).
DIALECT_ACOUSTICS: dict[
    str, tuple[float, tuple[float, float, float], float, str, tuple[str, ...]]
] = {
    "podhale": (110.0, (700.0, 1150.0, 2550.0), 0.72, "Podhale", ("małopolskie",)),
    "silesia": (150.0, (520.0, 1550.0, 2500.0), 0.60, "Górny Śląsk", ("śląskie", "opolskie")),
    "kashubia": (185.0, (420.0, 1950.0, 2750.0), 0.48, "Kaszuby", ("pomorskie",)),
    "mazovia_proper": (
        225.0,
        (360.0, 2350.0, 2950.0),
        0.34,
        "Mazowsze",
        ("mazowieckie", "łódzkie"),
    ),
}

#: Every catalogued class key, in the fixed sorted order the generator consumes
#: the RNG in (sorted class keys -> speaker index -> sample index). Materialised
#: here so the ordering guarantee lives with the data, not the logic.
DIALECTS: tuple[str, ...] = tuple(sorted(DIALECT_ACOUSTICS))

#: -3 dB bandwidths (Hz) of the F1/F2/F3 formant resonators, shared by every
#: class. Widening with formant index mirrors real speech, where higher formants
#: are more heavily damped.
FORMANT_BANDWIDTHS: tuple[float, float, float] = (80.0, 100.0, 120.0)

#: Linear amplitude weights of the F1/F2/F3 resonators in the parallel filter
#: bank. F1 dominates; higher formants contribute progressively less energy.
FORMANT_WEIGHTS: tuple[float, float, float] = (1.0, 0.6, 0.35)

#: Depth (fraction of F0) and rate (Hz) of the slow sinusoidal vibrato applied
#: to every clip, so the pitch track has a small non-zero standard deviation
#: rather than the dead-flat contour of a fixed-period pulse train.
VIBRATO_DEPTH: float = 0.02
VIBRATO_RATE_HZ: float = 5.0

#: RMS of the additive white aspiration noise, as a fraction of the voiced
#: signal's RMS. Kept low so probabilistic-YIN still locks onto F0, but non-zero
#: so formant LPC never sees a perfectly periodic (degenerate) spectrum.
NOISE_RMS_FRACTION: float = 0.03

#: Target RMS every clip is normalised to before int16 quantisation. Loudness is
#: held constant across classes on purpose: a systematic loudness difference
#: would be a trivial giveaway rather than a dialect signal. The value is low
#: enough that the peak of a formant-shaped pulse train stays well inside the
#: int16 range, so quantisation never clips.
TARGET_RMS: float = 0.1
