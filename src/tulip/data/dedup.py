"""Exact and near-duplicate removal for text corpora.

Web-derived dialect corpora repeat material heavily (the same folk tale
transcribed twice, the same NKJP sentence sampled from two subcorpora), and
duplicates that straddle a train/test boundary silently inflate scores. This
module removes them deterministically in two passes:

1. **Exact pass**: a stable hash of the normalised text (NFC, casefolded,
   whitespace-collapsed). Catches re-encodings and trivial reformatting.
2. **Near pass**: character 5-shingle Jaccard similarity, estimated with
   MinHash and made sub-quadratic with locality-sensitive banding: only
   texts sharing at least one signature band are compared, and candidate
   pairs are then *verified* with the exact Jaccard on shingle sets, so
   banding affects cost, never correctness of a drop decision. With the
   default 128 permutations in 32 bands, a true pair at the default
   threshold (0.85) is caught with probability > 0.999.

The default threshold of ``0.85`` was chosen for short dialect texts: for a
~200-character utterance it tolerates roughly a one-word edit (an inflection
change, a filler word) while keeping genuinely distinct sentences that merely
share formulaic openings. Raise it toward 0.95 to drop only near-verbatim
copies; lower it toward 0.7 for aggressive boilerplate removal.

Everything is pure stdlib + numpy, deterministic for a given ``seed`` and
input order (first occurrence wins), and streams in O(n) memory for ~1M
short texts.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from tulip.core.exceptions import ConfigurationError
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tulip.core.types import Sample

_logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")

#: Modulus for MinHash permutations (Mersenne prime 2**61 - 1). Shingle
#: hashes and coefficients are < 2**32, so ``a * h + b`` stays below 2**64
#: and uint64 arithmetic never overflows.
_MERSENNE_61 = np.uint64((1 << 61) - 1)


def normalise_for_hash(text: str) -> str:
    """Canonicalise text for duplicate detection (NFC, casefold, single spaces).

    Deliberately more aggressive than :class:`~tulip.data.cleaning.TextCleaner`:
    case and spacing differences should not protect a duplicate. Diacritics
    are preserved: two texts differing only in diacritics are genuinely
    different dialect data.
    """
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def text_fingerprint(text: str) -> str:
    """Return a stable hex fingerprint of the normalised text."""
    digest = hashlib.blake2b(normalise_for_hash(text).encode("utf-8"), digest_size=16)
    return digest.hexdigest()


def _shingle_hashes(normalised: str, shingle_size: int) -> frozenset[int]:
    """Return the set of stable 32-bit hashes of character shingles.

    Texts shorter than one shingle contribute themselves as a single
    shingle, so very short texts still participate in the near pass.
    ``zlib.crc32`` is used because it is stable across processes/platforms
    (unlike ``hash``) and fast; 32-bit collisions are negligible for a
    similarity estimate.

    Note:
        A numpy-vectorised rolling-hash replacement was tried and *measured
        slower* for typical short dialect texts (ufunc/`np.unique` per-call
        overhead exceeds ~80 crc32 calls), as was array-based Jaccard
        verification (`np.intersect1d` loses badly to C-level set ops for
        many small comparisons). Don't re-attempt without profiling first.
    """
    if len(normalised) <= shingle_size:
        shingles = {normalised} if normalised else set()
    else:
        shingles = {
            normalised[i : i + shingle_size] for i in range(len(normalised) - shingle_size + 1)
        }
    return frozenset(zlib.crc32(s.encode("utf-8")) for s in shingles)


def shingle_jaccard(a: str, b: str, *, shingle_size: int = 5) -> float:
    """Exact character-shingle Jaccard similarity between two raw texts."""
    sa = _shingle_hashes(normalise_for_hash(a), shingle_size)
    sb = _shingle_hashes(normalise_for_hash(b), shingle_size)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class _MinHasher:
    """MinHash signatures over 32-bit shingle hashes via universal hashing."""

    def __init__(self, num_permutations: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, 1 << 32, size=num_permutations, dtype=np.uint64)
        self._b = rng.integers(0, 1 << 32, size=num_permutations, dtype=np.uint64)

    def signature(self, shingles: frozenset[int]) -> np.ndarray:
        """Return the MinHash signature (uint64 vector) for a shingle set."""
        hashes = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
        products = hashes[:, None] * self._a[None, :] + self._b[None, :]
        return (products % _MERSENNE_61).min(axis=0)


@dataclass(frozen=True)
class DedupResult:
    """Outcome of deduplication: survivors plus the IDs of dropped samples."""

    samples: list[Sample]
    dropped_exact: list[str] = field(default_factory=list)
    dropped_near: list[str] = field(default_factory=list)

    @property
    def num_dropped(self) -> int:
        """Total number of removed samples."""
        return len(self.dropped_exact) + len(self.dropped_near)


def deduplicate_samples(
    samples: Iterable[Sample],
    *,
    shingle_size: int = 5,
    threshold: float = 0.85,
    num_permutations: int = 128,
    bands: int = 32,
    seed: int = 0,
    near_duplicates: bool = True,
) -> DedupResult:
    """Remove exact and near-duplicate texts, keeping first occurrences.

    Samples without text (audio-only) always survive: text hashing cannot
    judge them, and audio-level dedup is out of scope here.

    Args:
        samples: Input samples; earlier samples win ties, so order matters
            and the result is deterministic for a fixed input order.
        shingle_size: Character shingle length for the near pass.
        threshold: Jaccard similarity at or above which a text is dropped as
            a near-duplicate (see module docstring for how to choose it).
        num_permutations: MinHash signature length; must be divisible by
            ``bands``.
        bands: Number of LSH bands (more bands = higher candidate recall,
            more verification work).
        seed: Seed for the MinHash permutations.
        near_duplicates: Set ``False`` to run only the exact pass.

    Returns:
        A :class:`DedupResult` with surviving samples in input order.

    Raises:
        ConfigurationError: if the LSH shape or threshold is invalid.
    """
    if not 0.0 < threshold <= 1.0:
        raise ConfigurationError(f"dedup threshold must be in (0, 1], got {threshold}")
    if num_permutations <= 0 or bands <= 0 or num_permutations % bands != 0:
        raise ConfigurationError(
            f"num_permutations ({num_permutations}) must be a positive multiple of bands ({bands})"
        )
    rows_per_band = num_permutations // bands
    hasher = _MinHasher(num_permutations, seed) if near_duplicates else None

    kept: list[Sample] = []
    dropped_exact: list[str] = []
    dropped_near: list[str] = []
    seen_fingerprints: set[str] = set()
    # LSH state over *kept* texts: band buckets -> indices into shingle_sets.
    buckets: dict[tuple[int, bytes], list[int]] = {}
    shingle_sets: list[frozenset[int]] = []

    for sample in samples:
        if sample.text is None or not sample.text.strip():
            kept.append(sample)
            continue

        fingerprint = text_fingerprint(sample.text)
        if fingerprint in seen_fingerprints:
            dropped_exact.append(sample.id)
            continue
        seen_fingerprints.add(fingerprint)

        if hasher is None:
            kept.append(sample)
            continue

        shingles = _shingle_hashes(normalise_for_hash(sample.text), shingle_size)
        signature = hasher.signature(shingles)
        band_keys = [
            (band, signature[band * rows_per_band : (band + 1) * rows_per_band].tobytes())
            for band in range(bands)
        ]
        candidates: set[int] = set()
        for key in band_keys:
            candidates.update(buckets.get(key, ()))

        duplicate_of: int | None = None
        for candidate in sorted(candidates):  # earliest kept text first
            other = shingle_sets[candidate]
            union = len(shingles | other)
            if union and len(shingles & other) / union >= threshold:
                duplicate_of = candidate
                break

        if duplicate_of is not None:
            dropped_near.append(sample.id)
            continue

        index = len(shingle_sets)
        shingle_sets.append(shingles)
        for key in band_keys:
            buckets.setdefault(key, []).append(index)
        kept.append(sample)

    if dropped_exact or dropped_near:
        _logger.info(
            "deduplication removed %d exact and %d near duplicates (%d kept)",
            len(dropped_exact),
            len(dropped_near),
            len(kept),
        )
    return DedupResult(samples=kept, dropped_exact=dropped_exact, dropped_near=dropped_near)


__all__ = [
    "DedupResult",
    "deduplicate_samples",
    "normalise_for_hash",
    "shingle_jaccard",
    "text_fingerprint",
]
