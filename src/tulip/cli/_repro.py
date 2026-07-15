"""Reproducibility gate: does the suite still regenerate the committed board?

The leaderboard's headline promise is that ``leaderboard.md`` and
``provenance.json`` regenerate byte-for-byte. Today that is only ever checked by
a human running a manual diff. This module makes it a function (and, via the
CLI, a CI job): regenerate the whole suite into a scratch directory and compare
the guaranteed artifacts against the committed ones. Any hidden nondeterminism,
a moved generator default, an environment shift, or a silent metric regression
then fails loudly instead of shipping.

Only the two *guaranteed* artifacts are compared. ``leaderboard.json`` is
deliberately excluded: it retains wall-clock timings and is documented as not
part of the byte-identical guarantee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.evaluation.leaderboard import (
    LEADERBOARD_JSON,
    load_suite,
    run_leaderboard,
    write_leaderboard,
    write_significance,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["EXCLUDED_ARTIFACTS", "verify_reproduction"]

#: Committed artifacts NOT byte-compared: ``leaderboard.json`` retains wall-clock
#: timings and is documented as outside the byte-identical guarantee.
EXCLUDED_ARTIFACTS = frozenset({LEADERBOARD_JSON})


def verify_reproduction(suite_path: Path, committed_root: Path, work_dir: Path) -> list[str]:
    """Regenerate ``suite_path`` and compare its artifacts with the committed ones.

    Regenerates the full artifact set (leaderboard + provenance + significance)
    into ``work_dir`` and byte-compares every committed file except
    :data:`EXCLUDED_ARTIFACTS`. A committed file that is missing from the fresh
    run, or differs, is reported as drift.

    Args:
        suite_path: The leaderboard suite YAML to run.
        committed_root: Directory that holds the committed ``<suite>/`` artifacts
            (the ``--out`` root, e.g. ``benchmarks/results``).
        work_dir: A scratch directory to regenerate into (its contents are the
            caller's to clean up).

    Returns:
        A list of human-readable drift descriptions. An empty list means the
        suite reproduced the committed board exactly.
    """
    suite = load_suite(suite_path)
    results = run_leaderboard(suite)
    fresh_dir = work_dir / suite.name
    write_leaderboard(results, fresh_dir, suite=suite)
    write_significance(results, fresh_dir)

    committed_dir = committed_root / suite.name
    if not committed_dir.is_dir():
        return [f"no committed artifacts directory at {committed_dir}"]

    guarded = sorted(
        path.name
        for path in committed_dir.glob("*")
        if path.is_file() and path.name not in EXCLUDED_ARTIFACTS
    )
    drift: list[str] = []
    for name in guarded:
        fresh = fresh_dir / name
        if not fresh.is_file():
            drift.append(f"{name}: committed artifact was not regenerated")
        elif (committed_dir / name).read_bytes() != fresh.read_bytes():
            drift.append(f"{name}: regenerated output differs from the committed artifact")
    return drift
