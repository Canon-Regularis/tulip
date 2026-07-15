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
    from collections.abc import Sequence
    from pathlib import Path

    from tulip.evaluation.benchmark import BenchmarkResult
    from tulip.evaluation.leaderboard import LeaderboardSuite

__all__ = ["EXCLUDED_ARTIFACTS", "reproduce_from_scratch", "verify_reproduction"]

#: Committed artifacts NOT byte-compared: ``leaderboard.json`` retains wall-clock
#: timings and is documented as outside the byte-identical guarantee.
EXCLUDED_ARTIFACTS = frozenset({LEADERBOARD_JSON})


def verify_reproduction(suite_path: Path, committed_root: Path, work_dir: Path) -> list[str]:
    """Regenerate ``suite_path`` and compare its artifacts with the committed ones.

    Regenerates the full artifact set (leaderboard + provenance + significance)
    into ``work_dir`` and byte-compares every committed file except
    :data:`EXCLUDED_ARTIFACTS`. A committed file that is missing from the fresh
    run, or differs, is reported as drift. Each config builds under its own
    declared ``output_dir``; use :func:`reproduce_from_scratch` to isolate that
    too.

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
    return _regenerate_and_compare(suite, results, committed_root, work_dir)


def reproduce_from_scratch(suite_path: Path, committed_root: Path, work_dir: Path) -> list[str]:
    """Reproduce the committed board in full isolation, then match it byte-for-byte.

    Stronger than :func:`verify_reproduction`: every build artifact (splits,
    trained models) is redirected under ``work_dir`` too, so the run reads and
    writes nothing outside it. That proves the committed board depends only on the
    committed source and configs, not on any pre-existing artifacts tree, which is
    exactly what a fresh checkout in a clean container does. The provenance records
    only content digests and seeds, never an output path, so an isolated run still
    reproduces the committed board byte-for-byte on the platform that produced it.

    Args:
        suite_path: The leaderboard suite YAML to run.
        committed_root: Directory holding the committed ``<suite>/`` artifacts.
        work_dir: A scratch directory for both the build tree and the artifacts.

    Returns:
        Drift descriptions; an empty list means an exact, isolated reproduction.
    """
    suite = load_suite(suite_path)
    build_dir = work_dir / "build"
    results = run_leaderboard(suite, output_dir=build_dir)
    return _regenerate_and_compare(suite, results, committed_root, work_dir, build_dir=build_dir)


def _regenerate_and_compare(
    suite: LeaderboardSuite,
    results: Sequence[BenchmarkResult],
    committed_root: Path,
    work_dir: Path,
    *,
    build_dir: Path | None = None,
) -> list[str]:
    """Write the fresh artifacts and byte-compare them with the committed set.

    ``build_dir`` is where the run built its splits, threaded to provenance so an
    isolated run reads its sizes and dataset digest from the isolated build rather
    than a stale artifacts tree.
    """
    fresh_dir = work_dir / suite.name
    write_leaderboard(results, fresh_dir, suite=suite, build_dir=build_dir)
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
