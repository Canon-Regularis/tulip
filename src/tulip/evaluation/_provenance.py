"""The leaderboard's provenance subsystem: the deterministic audit record.

``provenance.json`` ties a committed leaderboard to exactly what produced it:
each config's seeds, its split sizes and class distribution, a content digest of
the dataset that fed it, the fixed-precision per-result metrics, and an
environment block. It is deterministic by construction (sorted keys, no
timestamps, no timings, fixed float precision), which is what lets the
reproducibility gate byte-compare it across runs.

Extracted from :mod:`tulip.evaluation.leaderboard` (which renders the human
board) so the audit-record assembly is its own unit, mirroring the sibling
:mod:`tulip.evaluation._provenance_env`. Key order and the ``round(x, 6)``
precision are load-bearing: they must not change, or a committed board stops
reproducing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tulip._serialize import tulip_version
from tulip.config.loader import load_experiment_config
from tulip.evaluation._provenance_env import environment_provenance
from tulip.utils.io import read_json
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.config.schemas import ExperimentConfig
    from tulip.evaluation.benchmark import BenchmarkResult
    from tulip.evaluation.leaderboard import LeaderboardSuite

__all__ = ["PROVENANCE_FLOAT_DIGITS", "PROVENANCE_JSON", "build_provenance"]

_logger = get_logger(__name__)

#: Artifact name for the provenance record.
PROVENANCE_JSON = "provenance.json"

#: Fixed rounding applied to every float in ``provenance.json`` so re-runs are
#: byte-identical even under trivial floating-point noise.
PROVENANCE_FLOAT_DIGITS = 6


def build_provenance(
    results: Sequence[BenchmarkResult],
    suite: LeaderboardSuite,
    *,
    split: str,
    build_dir: Path | None = None,
) -> dict[str, Any]:
    """Assemble the deterministic provenance payload for a leaderboard run.

    Args:
        results: The benchmark results the board was rendered from.
        suite: The leaderboard declaration (name, configs, models).
        split: The split whose reports feed the board (usually ``"test"``).
        build_dir: When set, the isolated build root the run wrote its splits
            under; provenance reads sizes/digests from there so a from-scratch
            reproduction never picks up a stale developer artifacts tree.

    Returns:
        The provenance mapping, ready for sorted-key JSON serialisation.
    """
    config_entries = [
        _config_provenance(config_path, load_experiment_config(config_path), build_dir=build_dir)
        for config_path in suite.configs
    ]
    result_entries = [
        entry for result in results if (entry := _result_provenance(result, split)) is not None
    ]
    # Total, seed-independent order: experiment, then best-first, then model.
    result_entries.sort(key=lambda e: (e["experiment"], -e["f1_macro"], e["model"]))
    return {
        "configs": config_entries,
        "environment": environment_provenance(suite.configs),
        "float_precision": PROVENANCE_FLOAT_DIGITS,
        "models": sorted(suite.models),
        "results": result_entries,
        "split": split,
        "suite": suite.name,
        "tulip_version": tulip_version(),
    }


def _config_provenance(
    config_path: Path, config: ExperimentConfig, *, build_dir: Path | None = None
) -> dict[str, Any]:
    """Provenance for one experiment: seeds, split sizes/distribution, and a data digest.

    The manifest and split lock are read from where the run actually built its
    splits: ``build_dir/<name>/splits`` for an isolated run, else the config's
    declared ``output_dir/<name>/splits``. Reading them from the build location
    keeps a from-scratch reproduction honest, rather than picking up a stale
    developer artifacts tree.
    """
    splits_dir = (
        (build_dir if build_dir is not None else config.output_dir) / config.name / "splits"
    )
    manifest = _read_manifest(splits_dir)
    return {
        "class_distribution": manifest.get("class_distribution") if manifest else None,
        "dataset_digest": _dataset_digest(splits_dir),
        "name": config.name,
        "path": Path(config_path).as_posix(),
        "seed": config.seed,
        "sizes": manifest.get("sizes") if manifest else None,
        "split_seed": config.split.seed,
    }


def _dataset_digest(splits_dir: Path) -> str | None:
    """The split lock's combined content digest under ``splits_dir``, or ``None``.

    Ties the leaderboard to the exact dataset content that fed it: the split
    lock's ``combined`` digest changes if any sample in any split is added,
    removed, or altered, so a silent data change is caught the same way an edited
    config or lexicon is. The digest is itself deterministic (an order-independent
    content hash), so it does not threaten the byte-stable guarantee.
    """
    from tulip.data.fingerprint import SPLIT_LOCK_NAME

    path = splits_dir / SPLIT_LOCK_NAME
    if not path.is_file():
        _logger.debug("no split lock at %s; dataset digest omitted from provenance", path)
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        # A corrupt or unreadable lock degrades to null, matching how an absent
        # build manifest degrades sizes: provenance never fails a valid run.
        _logger.debug("split lock %s unreadable; dataset digest omitted: %s", path, exc)
        return None
    if isinstance(data, dict) and data.get("combined") is not None:
        return str(data["combined"])
    return None


def _result_provenance(result: BenchmarkResult, split: str) -> dict[str, Any] | None:
    """Fixed-precision metrics for one result, or ``None`` if it lacks ``split``."""
    report = result.reports.get(split)
    if report is None:
        _logger.debug("result %r has no %r split; omitted from provenance", result.model, split)
        return None
    auc = report.roc_auc_macro_ovr
    calibration = report.calibration
    return {
        "accuracy": round(report.accuracy, PROVENANCE_FLOAT_DIGITS),
        "brier": None if calibration is None else round(calibration.brier, PROVENANCE_FLOAT_DIGITS),
        "ece": None if calibration is None else round(calibration.ece, PROVENANCE_FLOAT_DIGITS),
        "experiment": result.experiment,
        "f1_macro": round(report.f1_macro, PROVENANCE_FLOAT_DIGITS),
        "f1_weighted": round(report.f1_weighted, PROVENANCE_FLOAT_DIGITS),
        "model": result.model,
        "n_train": result.n_train,
        "roc_auc": None if auc is None else round(auc, PROVENANCE_FLOAT_DIGITS),
    }


def _read_manifest(splits_dir: Path) -> dict[str, Any] | None:
    """Read the ``build_manifest.json`` under ``splits_dir``, or ``None`` if absent.

    The manifest is written by :meth:`tulip.data.builder.DatasetBuilder.build`
    into the run's ``splits/`` directory; it is absent until the suite is actually
    run, so provenance degrades to ``null`` sizes rather than failing.
    """
    from tulip.data.builder import BUILD_MANIFEST_NAME

    path = splits_dir / BUILD_MANIFEST_NAME
    if not path.is_file():
        _logger.debug("no build manifest at %s; provenance sizes omitted", path)
        return None
    manifest = read_json(path)
    return manifest if isinstance(manifest, dict) else None
