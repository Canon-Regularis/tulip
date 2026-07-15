"""Content fingerprints for produced splits: prove a split really reproduced.

The build manifest records split *sizes* and class distribution, but not their
*content*. So the headline "byte-for-byte reproducible, speaker-disjoint split"
claim is unverifiable: a library upgrade that reorders deduplication, or a
generator default that shifts a boundary, changes which samples land where while
leaving every count identical, and nothing notices.

This module closes that gap with a canonical, order-independent BLAKE2b digest
over each split's membership:

* each sample is hashed from its canonical JSON (sorted keys), so the digest
  depends on content, not field order;
* a split's digest is the hash of its *sorted* per-sample digests, so it is
  invariant to incidental row order but sensitive to any membership change;
* :func:`verify_splits` recomputes and raises :class:`~tulip.core.exceptions.DataError`
  naming exactly which split drifted, so a regression fails loudly instead of
  shipping.

Committed as ``split_lock.json`` (deterministic: sorted keys, no timestamps), it
is CI-gateable. This module stays import-light: stdlib + pydantic + core types,
no numpy/sklearn, so ``import tulip.data`` keeps its lean footprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import write_sorted_json
from tulip.core.exceptions import DataError
from tulip.utils.io import read_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.core.types import Sample
    from tulip.data.splitting import DatasetSplits

__all__ = [
    "SPLIT_LOCK_NAME",
    "SplitFingerprint",
    "fingerprint_splits",
    "sample_digest",
    "split_digest",
    "verify_splits",
]

#: File name for a committed split lock.
SPLIT_LOCK_NAME = "split_lock.json"

_ALGORITHM = "blake2b-128"
_DIGEST_BYTES = 16


class SplitFingerprint(BaseModel):
    """A content fingerprint of one train/validation/test partition.

    Attributes:
        algorithm: The digest algorithm (e.g. ``blake2b-128``).
        sizes: Sample count per split.
        digests: Order-independent content digest per split.
        combined: A single digest over all splits, for a one-line equality check.
    """

    model_config = ConfigDict(frozen=True)

    algorithm: str
    sizes: dict[str, int]
    digests: dict[str, str]
    combined: str = Field(min_length=1)

    def save(self, path: Path | str) -> None:
        """Write the fingerprint as deterministic JSON (sorted keys)."""
        write_sorted_json(Path(path), self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: Path | str) -> SplitFingerprint:
        """Read a fingerprint written by :meth:`save`.

        Raises:
            DataError: if the file is not a split lock.
        """
        data = read_json(Path(path))
        if not isinstance(data, dict) or "combined" not in data or "digests" not in data:
            raise DataError(f"{path} is not a tulip split lock (expected 'digests' and 'combined')")
        return cls.model_validate(data)


def sample_digest(sample: Sample) -> str:
    """Canonical content digest of one sample (order-independent JSON)."""
    canonical = json.dumps(sample.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return _digest(canonical.encode("utf-8"))


def split_digest(samples: Sequence[Sample]) -> str:
    """Order-independent content digest of one split's samples.

    Hashes the *sorted* per-sample digests, so reordering the rows leaves the
    digest unchanged while any added, removed, or altered sample changes it.
    """
    joined = "\n".join(sorted(sample_digest(sample) for sample in samples))
    return _digest(joined.encode("utf-8"))


def fingerprint_splits(splits: DatasetSplits) -> SplitFingerprint:
    """Compute the :class:`SplitFingerprint` of a built partition."""
    digests = {name: split_digest(samples) for name, samples in splits.as_dict().items()}
    joined = "\n".join(f"{name}:{digest}" for name, digest in sorted(digests.items()))
    combined = _digest(joined.encode("utf-8"))
    return SplitFingerprint(
        algorithm=_ALGORITHM,
        sizes=splits.sizes(),
        digests=digests,
        combined=combined,
    )


def verify_splits(splits: DatasetSplits, expected: SplitFingerprint) -> None:
    """Check that ``splits`` reproduce ``expected``, raising on any drift.

    Args:
        splits: The freshly built partition.
        expected: The committed fingerprint to reproduce.

    Raises:
        DataError: naming every split whose size or content digest differs, so a
            non-reproducible split fails loudly rather than shipping silently.
    """
    actual = fingerprint_splits(splits)
    if actual.combined == expected.combined:
        return
    drift = [
        f"{name}: expected {expected.digests.get(name, '<missing>')[:12]} "
        f"(n={expected.sizes.get(name, '?')}) but got {actual.digests[name][:12]} "
        f"(n={actual.sizes[name]})"
        for name in actual.digests
        if actual.digests[name] != expected.digests.get(name)
    ]
    raise DataError(
        "split fingerprint mismatch - the split did not reproduce; "
        + "; ".join(drift or ["combined digest differs"])
    )


def _digest(data: bytes) -> str:
    """BLAKE2b hex digest at the module's fixed width."""
    return hashlib.blake2b(data, digest_size=_DIGEST_BYTES).hexdigest()
