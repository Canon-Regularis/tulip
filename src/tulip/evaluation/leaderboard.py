"""Reproducible public leaderboard: a byte-for-byte regenerable ranking.

A leaderboard is the project's headline deliverable -- a single, comparable
ranking of models on identical frozen splits that anyone can regenerate and
diff. This module is a thin, deterministic layer over the existing benchmark
machinery:

* :func:`run_leaderboard` loops a declarative :class:`LeaderboardSuite` through
  the untouched :func:`tulip.pipeline.experiment.run_benchmark`, so every model
  in every experiment is trained on the same speaker-disjoint split.
* :func:`render_leaderboard_markdown` reuses
  :func:`tulip.evaluation.benchmark.comparison_table` for sorting and NaN
  handling but emits *only* deterministic columns. It deliberately drops
  ``wall_seconds`` (wall-clock time is machine dependent and unfit for a
  committed artifact), which is the sole reason it exists instead of calling
  :func:`~tulip.evaluation.benchmark.to_markdown_table`.
* :func:`write_leaderboard` persists ``leaderboard.md`` (deterministic),
  ``leaderboard.json`` (the full raw dump, timings included), and
  ``provenance.json`` (deterministic: sorted keys, no timestamps, no timings).

The reproducibility guarantee: for a fixed seed and environment, ``leaderboard.md``
and ``provenance.json`` are byte-identical across re-runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tulip.config.loader import load_experiment_config
from tulip.core.exceptions import ConfigurationError
from tulip.evaluation._format import format_metric, markdown_table, write_sorted_json
from tulip.evaluation._provenance_env import environment_provenance
from tulip.evaluation.benchmark import BenchmarkResult, save_benchmark
from tulip.utils.io import read_json, read_yaml
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.config.schemas import ExperimentConfig

_logger = get_logger(__name__)

#: Split whose reports feed the committed leaderboard and provenance.
DEFAULT_SPLIT = "test"
#: File names written by :func:`write_leaderboard`.
LEADERBOARD_MD = "leaderboard.md"
LEADERBOARD_JSON = "leaderboard.json"
PROVENANCE_JSON = "provenance.json"
#: Fixed rounding applied to every float in ``provenance.json`` so re-runs are
#: byte-identical even under trivial floating-point noise.
PROVENANCE_FLOAT_DIGITS = 6

#: The deterministic leaderboard columns; a strict subset of the benchmark
#: table that excludes the nondeterministic ``wall_seconds``. ``Experiment`` is
#: carried because a suite runs each competitor model against *every* config,
#: so ``Model`` alone does not identify a row.
_LEADERBOARD_HEADERS = (
    "Experiment",
    "Model",
    "Accuracy",
    "F1 (macro)",
    "F1 (weighted)",
    "ROC AUC",
    "ECE",
    "Brier",
    "Train",
)

__all__ = [
    "DEFAULT_SPLIT",
    "LEADERBOARD_JSON",
    "LEADERBOARD_MD",
    "PROVENANCE_JSON",
    "LeaderboardSuite",
    "load_suite",
    "render_leaderboard_markdown",
    "run_leaderboard",
    "write_leaderboard",
    "write_significance",
]


class LeaderboardSuite(BaseModel):
    """Declarative description of one reproducible leaderboard.

    Owned by this module rather than layered onto
    :class:`~tulip.config.schemas.ExperimentConfig`: a suite is a *collection*
    of experiments plus the competitor set applied to each, which is a
    different concept from a single experiment declaration.

    Attributes:
        name: Human-readable suite identifier, echoed into ``provenance.json``.
        configs: Experiment YAMLs; each is loaded and handed to
            :func:`~tulip.pipeline.experiment.run_benchmark`. Relative paths are
            resolved against the process working directory (run from the repo
            root, as documented in ``benchmarks/README.md``).
        models: Competitor model registry names applied to *every* config. When
            empty, each config's own ``model`` entry is the sole competitor.
        calibration_bins: When set, every model's report gains a top-label
            calibration block (ECE/MCE/Brier) with this many uniform bins, and
            the leaderboard shows ECE and Brier columns. ``None`` (the default)
            leaves calibration off and those columns render ``n/a``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    configs: list[Path]
    models: list[str] = Field(default_factory=list)
    calibration_bins: int | None = Field(default=None, ge=1)


def load_suite(path: Path | str) -> LeaderboardSuite:
    """Load and validate a :class:`LeaderboardSuite` from a YAML file.

    Args:
        path: The suite YAML (e.g. ``benchmarks/suite.yaml``).

    Returns:
        The validated suite.

    Raises:
        ConfigurationError: If the file is missing, unparsable, not a mapping,
            or fails schema validation.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"leaderboard suite file not found: {path}")
    try:
        raw = read_yaml(path)
    except Exception as exc:
        raise ConfigurationError(f"could not parse YAML suite {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"suite {path} must be a YAML mapping, got {type(raw).__name__}")
    try:
        return LeaderboardSuite.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid leaderboard suite {path}:\n{exc}") from exc


def run_leaderboard(suite: LeaderboardSuite) -> list[BenchmarkResult]:
    """Run every ``(config, model)`` pair in ``suite`` on its frozen split.

    Each config is loaded and benchmarked with the identical, untouched
    :func:`~tulip.pipeline.experiment.run_benchmark`, so competitor models see
    byte-identical speaker-disjoint splits. Results are concatenated in suite
    order; each carries its originating experiment name for later
    disambiguation in ``leaderboard.json`` / ``provenance.json``.

    Args:
        suite: The leaderboard declaration.

    Returns:
        One :class:`BenchmarkResult` per ``(config, model)`` pair.
    """
    # Imported lazily: the pipeline pulls in the training stack, which the pure
    # rendering/persistence helpers in this module do not need.
    from tulip.pipeline.experiment import run_benchmark

    models = list(suite.models) or None
    results: list[BenchmarkResult] = []
    for config_path in suite.configs:
        config = load_experiment_config(config_path)
        _logger.info("leaderboard %r: benchmarking config %r", suite.name, config.name)
        results.extend(run_benchmark(config, models, calibration_bins=suite.calibration_bins))
    _logger.info("leaderboard %r: produced %d result(s)", suite.name, len(results))
    return results


def render_leaderboard_markdown(
    results: Sequence[BenchmarkResult], *, split: str = DEFAULT_SPLIT
) -> str:
    """Render the deterministic leaderboard table as GitHub-flavoured markdown.

    Rows are ranked by descending macro F1, with ties broken by
    ``(experiment, model)`` so the ordering is total and independent of the
    order ``results`` arrives in -- a committed artifact must not change because
    a suite listed its configs differently.

    It never emits ``wall_seconds``: wall-clock time is machine dependent and
    would make the artifact non-regenerable.

    This deliberately does *not* delegate to
    :func:`~tulip.evaluation.benchmark.comparison_table`. That table identifies
    a row by ``model`` alone, which collides here: a suite trains every
    competitor against every config, so the same model name appears once per
    experiment and the rows would be indistinguishable.

    Args:
        results: The benchmark results to rank.
        split: Which split's reports to read from each result.

    Returns:
        A markdown table; unavailable ROC AUC renders as ``n/a``.
    """
    ranked = sorted(
        results,
        key=lambda result: (
            -result.report_for(split).f1_macro,
            result.experiment,
            result.model,
        ),
    )
    rows = []
    for result in ranked:
        report = result.report_for(split)
        calibration = report.calibration
        rows.append(
            (
                result.experiment,
                result.model,
                format_metric(report.accuracy),
                format_metric(report.f1_macro),
                format_metric(report.f1_weighted),
                format_metric(report.roc_auc_macro_ovr),
                format_metric(None if calibration is None else calibration.ece),
                format_metric(None if calibration is None else calibration.brier),
                str(result.n_train),
            )
        )
    return markdown_table(_LEADERBOARD_HEADERS, rows)


def write_leaderboard(
    results: Sequence[BenchmarkResult], out_dir: Path | str, *, suite: LeaderboardSuite
) -> None:
    """Write the leaderboard artifacts into ``out_dir``.

    Emits three files:

    * ``leaderboard.md`` -- the deterministic ranking table.
    * ``leaderboard.json`` -- the full raw benchmark dump (via
      :func:`~tulip.evaluation.benchmark.save_benchmark`); it retains raw
      per-model timings for reference and is therefore *not* part of the
      byte-identical guarantee.
    * ``provenance.json`` -- deterministic audit record with sorted keys, no
      timestamps and no timings: ``tulip_version``, per-config seed and split
      seed, the competitor model list, per-config split sizes and class
      distribution read from each run's ``build_manifest.json`` (``null`` when
      the manifest is absent), fixed-precision per-result metrics (including
      ECE/Brier when calibration is enabled) tagged with their experiment, and
      an ``environment`` block (Python floor, key dependency versions from
      ``uv.lock``, and content digests of the configs and shipped lexicons).

    Args:
        results: The benchmark results to publish.
        out_dir: Directory to write into (created if needed).
        suite: The suite that produced ``results`` (its configs are re-read for
            provenance).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_leaderboard_markdown(results)
    (out_dir / LEADERBOARD_MD).write_text(markdown + "\n", encoding="utf-8", newline="\n")

    save_benchmark(results, out_dir / LEADERBOARD_JSON)

    provenance = _build_provenance(results, suite)
    write_sorted_json(out_dir / PROVENANCE_JSON, provenance)
    _logger.info("wrote leaderboard artifacts to %s", out_dir)


def write_significance(
    results: Sequence[BenchmarkResult],
    out_dir: Path | str,
    *,
    split: str = DEFAULT_SPLIT,
    seed: int = 0,
) -> list[str]:
    """Write per-experiment significance artifacts from the in-memory predictions.

    Models within one experiment are trained on the identical frozen split, so
    their per-sample predictions are paired: this groups results by experiment
    and, for any experiment with at least two models carrying predictions, writes
    ``significance-<experiment>.md`` and ``significance-<experiment>.json`` (both
    deterministic). Experiments with predictions absent (e.g. a reloaded
    ``leaderboard.json``, whose predictions are excluded) are skipped.

    Args:
        results: The benchmark results, with ``predictions`` populated.
        out_dir: Directory to write into.
        split: Which split's predictions to test.
        seed: Bootstrap seed for the confidence intervals.

    Returns:
        The experiment names for which a report was written, in sorted order.
    """
    from tulip.evaluation.significance import paired_significance

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_experiment: dict[str, list[Any]] = {}
    for result in results:
        predictions = result.predictions.get(split)
        if predictions is not None:
            by_experiment.setdefault(result.experiment, []).append(predictions)

    written: list[str] = []
    for experiment, experiment_predictions in sorted(by_experiment.items()):
        if len(experiment_predictions) < 2:
            continue
        report = paired_significance(experiment_predictions, seed=seed)
        report.save(out_dir / f"significance-{experiment}.json")
        (out_dir / f"significance-{experiment}.md").write_text(
            report.to_markdown() + "\n", encoding="utf-8", newline="\n"
        )
        written.append(experiment)
    if written:
        _logger.info("wrote significance for %d experiment(s) to %s", len(written), out_dir)
    return written


def _build_provenance(
    results: Sequence[BenchmarkResult], suite: LeaderboardSuite
) -> dict[str, Any]:
    """Assemble the deterministic provenance payload (see :func:`write_leaderboard`)."""
    config_entries = [
        _config_provenance(config_path, load_experiment_config(config_path))
        for config_path in suite.configs
    ]
    result_entries = [
        entry
        for result in results
        if (entry := _result_provenance(result, DEFAULT_SPLIT)) is not None
    ]
    # Total, seed-independent order: experiment, then best-first, then model.
    result_entries.sort(key=lambda e: (e["experiment"], -e["f1_macro"], e["model"]))
    return {
        "configs": config_entries,
        "environment": environment_provenance(suite.configs),
        "float_precision": PROVENANCE_FLOAT_DIGITS,
        "models": sorted(suite.models),
        "results": result_entries,
        "split": DEFAULT_SPLIT,
        "suite": suite.name,
        "tulip_version": _tulip_version(),
    }


def _config_provenance(config_path: Path, config: ExperimentConfig) -> dict[str, Any]:
    """Provenance for one experiment: seeds plus split sizes/distribution."""
    manifest = _read_manifest(config)
    return {
        "class_distribution": manifest.get("class_distribution") if manifest else None,
        "name": config.name,
        "path": Path(config_path).as_posix(),
        "seed": config.seed,
        "sizes": manifest.get("sizes") if manifest else None,
        "split_seed": config.split.seed,
    }


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


def _read_manifest(config: ExperimentConfig) -> dict[str, Any] | None:
    """Read a run's ``build_manifest.json``, or ``None`` if it is not on disk.

    The manifest is written by :meth:`tulip.data.builder.DatasetBuilder.build`
    under ``<output_dir>/<name>/splits/``; it is absent until the suite is
    actually run, so provenance degrades to ``null`` sizes rather than failing.
    """
    from tulip.data.builder import BUILD_MANIFEST_NAME

    path = config.output_dir / config.name / "splits" / BUILD_MANIFEST_NAME
    if not path.is_file():
        _logger.debug("no build manifest at %s; provenance sizes omitted", path)
        return None
    manifest = read_json(path)
    return manifest if isinstance(manifest, dict) else None


def _tulip_version() -> str:
    """Return the installed tulip version (dev fallback handled by the package)."""
    import tulip

    return getattr(tulip, "__version__", "unknown")
