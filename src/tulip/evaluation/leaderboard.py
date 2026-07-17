"""Reproducible public leaderboard: a byte-for-byte regenerable ranking.

A leaderboard is the project's headline deliverable: a single, comparable
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

from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import write_markdown
from tulip.config.loader import load_experiment_config
from tulip.evaluation._format import format_metric, markdown_table, write_sorted_json
from tulip.evaluation._provenance import PROVENANCE_JSON, build_provenance
from tulip.evaluation._suite_loader import load_yaml_model
from tulip.evaluation.benchmark import BenchmarkResult, save_benchmark
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence


_logger = get_logger(__name__)

#: Split whose reports feed the committed leaderboard and provenance.
DEFAULT_SPLIT = "test"
#: File names written by :func:`write_leaderboard`.
LEADERBOARD_MD = "leaderboard.md"
LEADERBOARD_JSON = "leaderboard.json"

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
    caption: str | None = Field(default=None)
    """An optional note rendered above the ``leaderboard.md`` table, for example to
    label a board as a synthetic fixture rather than real accuracy. Presentation
    only: it never enters ``provenance.json``."""


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
    return load_yaml_model(path, LeaderboardSuite, noun="leaderboard suite")


def run_leaderboard(
    suite: LeaderboardSuite, *, output_dir: Path | None = None, n_jobs: int = 1
) -> list[BenchmarkResult]:
    """Run every ``(config, model)`` pair in ``suite`` on its frozen split.

    Each config is loaded and benchmarked with the identical, untouched
    :func:`~tulip.pipeline.experiment.run_benchmark`, so competitor models see
    byte-identical speaker-disjoint splits. Results are concatenated in suite
    order; each carries its originating experiment name for later
    disambiguation in ``leaderboard.json`` / ``provenance.json``.

    Args:
        suite: The leaderboard declaration.
        output_dir: When set, overrides every config's ``output_dir`` so all
            build artifacts (splits, models) land here instead of each config's
            declared tree. A from-scratch reproduction passes a throwaway
            directory, so the run depends only on the committed source and never
            reads or writes the developer's own artifacts tree.
        n_jobs: Competitors to train in parallel within each config (passed to
            :func:`~tulip.pipeline.experiment.run_benchmark`). ``1`` (the
            default) is in-process and sequential, so the committed board is
            byte-for-byte unchanged; above 1 trains in separate processes with
            identical results.

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
        if output_dir is not None:
            config = config.model_copy(update={"output_dir": Path(output_dir)})
        _logger.info("leaderboard %r: benchmarking config %r", suite.name, config.name)
        results.extend(
            run_benchmark(config, models, calibration_bins=suite.calibration_bins, n_jobs=n_jobs)
        )
    _logger.info("leaderboard %r: produced %d result(s)", suite.name, len(results))
    return results


def render_leaderboard_markdown(
    results: Sequence[BenchmarkResult], *, split: str = DEFAULT_SPLIT, caption: str | None = None
) -> str:
    """Render the deterministic leaderboard table as GitHub-flavoured markdown.

    Rows are ranked by descending macro F1, with ties broken by
    ``(experiment, model)`` so the ordering is total and independent of the
    order ``results`` arrives in: a committed artifact must not change because
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
        caption: An optional note rendered as a preamble above the table (e.g. a
            synthetic-fixture disclaimer). Deterministic, so it stays byte-stable.

    Returns:
        A markdown table (with the optional caption above it); unavailable ROC AUC
        renders as ``n/a``.
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
    table = markdown_table(_LEADERBOARD_HEADERS, rows)
    if caption and caption.strip():
        return f"{caption.strip()}\n\n{table}"
    return table


def write_leaderboard(
    results: Sequence[BenchmarkResult],
    out_dir: Path | str,
    *,
    suite: LeaderboardSuite,
    build_dir: Path | None = None,
) -> None:
    """Write the leaderboard artifacts into ``out_dir``.

    Emits three files:

    * ``leaderboard.md``: the deterministic ranking table.
    * ``leaderboard.json``: the full raw benchmark dump (via
      :func:`~tulip.evaluation.benchmark.save_benchmark`); it retains raw
      per-model timings for reference and is therefore *not* part of the
      byte-identical guarantee.
    * ``provenance.json``: deterministic audit record with sorted keys, no
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
        build_dir: Where the run built its splits, when that differs from each
            config's declared ``output_dir`` (an isolated from-scratch run passes
            the throwaway build root here). Provenance reads the per-split sizes
            and dataset digest from ``build_dir/<name>/splits`` instead, so the
            audit record reflects the splits that actually fed the run rather than
            a stale developer artifacts tree. ``None`` uses each config's declared
            ``output_dir``, the normal case.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_leaderboard_markdown(results, caption=suite.caption)
    write_markdown(out_dir / LEADERBOARD_MD, markdown)

    save_benchmark(results, out_dir / LEADERBOARD_JSON)

    provenance = build_provenance(results, suite, split=DEFAULT_SPLIT, build_dir=build_dir)
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
        write_markdown(out_dir / f"significance-{experiment}.md", report.to_markdown())
        written.append(experiment)
    if written:
        _logger.info("wrote significance for %d experiment(s) to %s", len(written), out_dir)
    return written
