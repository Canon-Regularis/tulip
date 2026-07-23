"""Procedural generator for a linguistically-grounded synthetic dialect corpus.

The corpus exists so the whole toolkit is exercisable end-to-end with *zero*
data acquisition, yet carries **real, learnable** dialect signal rather than
random noise. Every generated text is a standard-Polish carrier sentence into
which two grounded sources of dialect information are injected:

* **Lexical**: a seeded subset of each dialect's genuine marker lexemes from
  ``tulip/features/text/lexicons/dialect_markers.yaml`` (e.g. Podhale ``baca``,
  Silesian ``gryfny``, Kashubian ``chëcz``), which gives the whole-word
  keyword/TF-IDF features something to key on.
* **Phonological**: deterministic string transforms reproducing real dialect
  processes: *mazurzenie* (cz/sz/ż/dż -> c/s/z/dz) for the Masovian group
  (Kurpie, Mazovia) and asynchronous soft-labial respelling (pi/bi/wi/mi ->
  psi/bzi/wzi/mni) for Kurpie. Applying these to the standard carrier
  reproduces the lexicon's own Kurpie forms (``piwo -> psiwo``,
  ``jeszcze -> jesce``, ``kobieta -> kobzieta``) and hands the character
  n-gram features signal that a whole-word lexicon cannot carry.

Three further design choices make the corpus behave like a real one:

* **Speaker idiolect**: each speaker draws a seeded personal filler
  vocabulary and marker subset, so a model can partly re-identify the speaker.
  That leakage is precisely what speaker-disjoint splitting must defend
  against, so the split step is genuinely exercised.
* **Cross-class noise** (``noise_level``): a small fraction of samples pick
  up a foreign marker (or, for the ``standard`` class, any marker), so the task
  is not trivially separable.
* **Marker dropout** (``marker_dropout``): a fraction of samples carry *no*
  lexical marker at all, mirroring the fact that plenty of real dialect
  utterances contain no diagnostic lexeme. Such a sample is recoverable only
  from a phonological transform (Kurpie, Mazovia) and is otherwise genuinely
  ambiguous. This is deliberate: without it every linear model scores a perfect
  1.000 and the leaderboard cannot rank anything. Dropout gives the task an
  irreducible error floor, which is what makes it a *benchmark* rather than a
  smoke test.

Generation is fully deterministic: a single ``numpy.random.default_rng(seed)``
is consumed in a fixed order (sorted dialects -> speaker index -> sample
index), so the same :class:`SyntheticSpec` always yields byte-identical
samples. The module is intentionally import-light (no sklearn/torch): the
lexicon loader is imported lazily inside :func:`generate_corpus`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import DialectLabels, Sample
from tulip.data._paths import ensure_directory
from tulip.data._synthetic_common import (
    base_manifest_record,
    resolve_dialect_keys,
    validate_common_spec,
)
from tulip.data._synthetic_corpus import ACTIONS as _ACTIONS
from tulip.data._synthetic_corpus import CARRIERS as _CARRIERS
from tulip.data._synthetic_corpus import FILLERS as _FILLERS
from tulip.data._synthetic_corpus import GEOGRAPHY as _GEOGRAPHY
from tulip.data._synthetic_corpus import MAZURZENIE as _MAZURZENIE
from tulip.data._synthetic_corpus import OBJECTS as _OBJECTS
from tulip.data._synthetic_corpus import PEOPLE as _PEOPLE
from tulip.data._synthetic_corpus import PLACES as _PLACES
from tulip.data._synthetic_corpus import SOFT_LABIALS as _SOFT_LABIALS
from tulip.data._synthetic_corpus import TIMES as _TIMES
from tulip.utils.io import write_jsonl
from tulip.utils.logging import get_logger

__all__ = ["SyntheticSpec", "generate_corpus", "write_synthetic_manifest"]

_logger = get_logger(__name__)

#: Value recorded in :attr:`Sample.source` for every generated sample.
SOURCE = "synthetic"

#: Sentinel used internally as the ``standard`` (non-dialectal) negative class.
#: It is deliberately not a lexicon key, so it can never collide with one.
_STANDARD = "standard"

#: Lexicon keys of the Masovian group, which undergo *mazurzenie*.
_MAZURZENIE_KEYS = frozenset({"kurpie", "masovia"})

#: Lexicon keys with asynchronous soft-labial respelling (Kurpie only).
_SOFT_LABIAL_KEYS = frozenset({"kurpie"})


@dataclass(frozen=True)
class SyntheticSpec:
    """Knobs controlling synthetic corpus generation.

    Args:
        n_speakers_per_dialect: Distinct speakers generated per class. Must be
            >= 2 so a speaker-disjoint split has more than one group to work
            with (the corpus's whole point).
        samples_per_speaker: Texts generated per speaker.
        dialects: Lexicon keys to include (``None`` selects every lexicon
            dialect). Keys are the lexicon's own (e.g. ``"masovia"``), not the
            taxonomy dialect values.
        include_standard: Add a ``standard`` negative class (carriers with no
            markers and no phonological transform).
        noise_level: Probability that a sample also receives a foreign marker,
            introducing cross-class leakage so the task is not trivial.
        marker_dropout: Probability that a sample carries no lexical marker at
            all. Raising it makes the task harder (and the benchmark
            discriminative); lowering it to 0.0 makes every class trivially
            separable and saturates every linear model at 1.000.
        seed: Seed for the single generator RNG; fixes the entire output.

    Raises:
        ConfigurationError: if any knob is out of range.
    """

    n_speakers_per_dialect: int = 8
    samples_per_speaker: int = 12
    dialects: tuple[str, ...] | None = None
    include_standard: bool = True
    noise_level: float = 0.10
    marker_dropout: float = 0.20
    seed: int = 7

    def __post_init__(self) -> None:
        validate_common_spec(self.n_speakers_per_dialect, self.samples_per_speaker, self.dialects)
        if not 0.0 <= self.noise_level <= 1.0:
            raise ConfigurationError(f"noise_level must be within [0, 1], got {self.noise_level}")
        if not 0.0 <= self.marker_dropout <= 1.0:
            raise ConfigurationError(
                f"marker_dropout must be within [0, 1], got {self.marker_dropout}"
            )


def generate_corpus(spec: SyntheticSpec) -> list[Sample]:
    """Generate the full synthetic corpus described by ``spec``.

    The output is deterministic for a given ``spec``: one RNG is consumed in a
    fixed order (sorted dialect keys, then speaker index, then sample index),
    so two calls with the same seed return identical ids, texts, and labels.

    Args:
        spec: Generation knobs (see :class:`SyntheticSpec`).

    Returns:
        The generated :class:`Sample` list, dialect classes first (sorted) then
        the optional ``standard`` class.

    Raises:
        ConfigurationError: if ``spec.dialects`` names an unknown lexicon key.
    """
    from tulip.features.text.keywords import canonical_dialect

    lexicon = _load_lexicon()
    all_keys = tuple(sorted(lexicon))
    selected = resolve_dialect_keys(spec.dialects, all_keys, kind="synthetic")
    #: Flat pool of every marker; used to inject cross-class noise.
    all_markers = tuple(sorted({marker for markers in lexicon.values() for marker in markers}))

    class_order: list[str] = list(selected)
    if spec.include_standard:
        class_order.append(_STANDARD)

    rng = np.random.default_rng(spec.seed)
    samples: list[Sample] = []
    for key in class_order:
        is_standard = key == _STANDARD
        markers = () if is_standard else lexicon[key]
        dialect_value = None if is_standard else canonical_dialect(key)
        region, voivodeship_pool = (None, ()) if is_standard else _GEOGRAPHY[key]

        for speaker_index in range(spec.n_speakers_per_dialect):
            speaker_id = f"{SOURCE}-{key}-spk{speaker_index:02d}"
            voivodeship = _choice(rng, voivodeship_pool) if voivodeship_pool else None
            speaker_fillers = _draw_fillers(rng)
            speaker_markers = _speaker_marker_subset(rng, markers)

            for sample_index in range(spec.samples_per_speaker):
                text = _make_text(
                    rng,
                    key=key,
                    markers=speaker_markers,
                    fillers=speaker_fillers,
                    noise_level=spec.noise_level,
                    marker_dropout=spec.marker_dropout,
                    foreign_markers=all_markers,
                )
                labels = (
                    DialectLabels(family="standard")
                    if is_standard
                    else DialectLabels(
                        dialect=dialect_value, region=region, voivodeship=voivodeship
                    )
                )
                samples.append(
                    Sample(
                        id=f"{SOURCE}-{key}-spk{speaker_index:02d}-{sample_index:03d}",
                        text=text,
                        speaker_id=speaker_id,
                        labels=labels,
                        source=SOURCE,
                        metadata={"generator": "tulip-synthetic", "spec_seed": spec.seed},
                    )
                )

    _logger.info(
        "generated %d synthetic samples across %d classes (seed=%d)",
        len(samples),
        len(class_order),
        spec.seed,
    )
    return samples


def write_synthetic_manifest(spec: SyntheticSpec, root: Path) -> Path:
    """Generate the corpus and persist it as ``root/manifest.jsonl``.

    The manifest is written in the flat, one-object-per-line shape that
    :func:`tulip.data.manifest.read_manifest` consumes, so a generated corpus
    can be checked in and re-loaded auditable-path instead of regenerated.

    Args:
        spec: Generation knobs.
        root: Directory to create the manifest under (created if absent).

    Returns:
        The path to the written ``manifest.jsonl``.
    """
    root = Path(root)
    root = ensure_directory(root, purpose="synthetic manifest")
    path = root / "manifest.jsonl"
    corpus = generate_corpus(spec)
    written = write_jsonl(path, (_to_manifest_record(sample) for sample in corpus))
    _logger.info("wrote %d synthetic samples to %s", written, path)
    return path


def _to_manifest_record(sample: Sample) -> dict[str, object]:
    """Flatten one :class:`Sample` into a read_manifest-compatible record."""
    return {
        "id": sample.id,
        "text": sample.text,
        "speaker_id": sample.speaker_id,
        **base_manifest_record(sample),
    }


def _make_text(
    rng: np.random.Generator,
    *,
    key: str,
    markers: tuple[str, ...],
    fillers: tuple[str, ...],
    noise_level: float,
    marker_dropout: float,
    foreign_markers: tuple[str, ...],
) -> str:
    """Assemble one sample's text: carrier + markers + idiolect + phonology."""
    template = _choice(rng, _CARRIERS)
    filled = template.format(
        place=_choice(rng, _PLACES),
        place2=_choice(rng, _PLACES),
        person=_choice(rng, _PEOPLE),
        person2=_choice(rng, _PEOPLE),
        action=_choice(rng, _ACTIONS),
        action2=_choice(rng, _ACTIONS),
        object=_choice(rng, _OBJECTS),
        object2=_choice(rng, _OBJECTS),
        time=_choice(rng, _TIMES),
    )

    parts = [filled]
    # The roll is drawn unconditionally so the RNG stream advances identically
    # for marker-bearing and standard classes, keeping generation order-stable.
    keeps_markers = rng.random() >= marker_dropout
    if markers and keeps_markers:
        # One or two markers, not a pile of them: a sample stuffed with every
        # marker of its class is separable by inspection and teaches nothing.
        count = min(int(rng.integers(1, 3)), len(markers))
        chosen = rng.choice(len(markers), size=count, replace=False)
        parts.append(" ".join(markers[i] for i in sorted(chosen)))
    if fillers:
        parts.append(" ".join(fillers))

    text = _transform(" ".join(parts), key)

    # Cross-class noise is appended *after* the phonological transform so a
    # foreign marker keeps its own dialect's spelling (it is not natively
    # subject to this class's sound changes).
    if foreign_markers and rng.random() < noise_level:
        text = f"{text} {_choice(rng, foreign_markers)}"
    return str(text)


def _transform(text: str, key: str) -> str:
    """Apply the phonological string transforms for ``key`` (if any)."""
    if key in _SOFT_LABIAL_KEYS:
        for src, dst in _SOFT_LABIALS:
            text = text.replace(src, dst)
    if key in _MAZURZENIE_KEYS:
        for src, dst in _MAZURZENIE:
            text = text.replace(src, dst)
    return text


def _speaker_marker_subset(rng: np.random.Generator, markers: tuple[str, ...]) -> tuple[str, ...]:
    """Drop 0-2 of a dialect's markers to give each speaker a slight idiolect.

    Enough markers are always retained (>= 2 whenever the dialect has that
    many) that the shared, generalisable dialect signal survives.
    """
    if len(markers) <= 3:
        return markers
    drop = int(rng.integers(0, 3))
    if drop == 0:
        return markers
    keep = len(markers) - drop
    indices = rng.choice(len(markers), size=keep, replace=False)
    return tuple(markers[i] for i in sorted(indices))


def _draw_fillers(rng: np.random.Generator) -> tuple[str, ...]:
    """Draw a speaker's personal 2-3 filler particles."""
    count = int(rng.integers(2, 4))
    indices = rng.choice(len(_FILLERS), size=count, replace=False)
    return tuple(_FILLERS[i] for i in sorted(indices))


def _choice(rng: np.random.Generator, pool: tuple[str, ...]) -> str:
    """Pick one element from ``pool`` as a plain ``str`` (not ``numpy.str_``)."""
    return str(pool[int(rng.integers(0, len(pool)))])


def _load_lexicon() -> dict[str, tuple[str, ...]]:
    """Load the bundled dialect-marker lexicon (lazy, to keep imports light)."""
    from tulip.features.text.keywords import load_lexicon

    return load_lexicon()
