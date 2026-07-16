"""Reproducible experiment and benchmark runners.

:func:`run_experiment` executes one fully declared experiment
(:class:`~tulip.config.schemas.ExperimentConfig`) end to end: seed, build
leakage-free splits, train, evaluate, and persist every artifact needed to
audit or reproduce the run. :func:`run_benchmark` trains several models over
the *identical* frozen split, the reproducible-benchmark deliverable: there
is currently no widely adopted benchmark for Polish dialect identification,
and comparable numbers require byte-identical splits.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from tulip._serialize import write_markdown
from tulip.config.loader import save_experiment_config
from tulip.config.schemas import ComponentConfig, ExperimentConfig
from tulip.core.exceptions import DataError
from tulip.data.builder import DatasetBuilder
from tulip.evaluation.benchmark import BenchmarkResult, save_benchmark, to_markdown_table
from tulip.evaluation.report import EvaluationReport
from tulip.pipeline.classifier import DialectClassifier
from tulip.pipeline.scoring import _evaluate_split, collect_predictions, evaluate_samples
from tulip.utils.io import write_json
from tulip.utils.logging import get_logger
from tulip.utils.seed import set_global_seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.data.splitting import DatasetSplits
    from tulip.evaluation.predictions import SplitPredictions

_logger = get_logger(__name__)


class ExperimentResult(BaseModel):
    """Everything a finished experiment produced (paths, sizes, reports)."""

    # protected_namespaces=(): "model_path" is a legitimate field name here.
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    name: str
    model: str
    task: str
    target: str
    sizes: dict[str, int]
    reports: dict[str, EvaluationReport]
    model_path: Path
    seconds: float

    def summary(self) -> str:
        """One line per evaluated split, for terminal output."""
        lines = [f"experiment {self.name!r} (model={self.model}, target={self.target})"]
        lines.extend(
            f"  {split}: {report.summary_line()}" for split, report in self.reports.items()
        )
        return "\n".join(lines)


def run_experiment(
    config: ExperimentConfig,
    *,
    splits: DatasetSplits | None = None,
    calibration_bins: int | None = None,
) -> ExperimentResult:
    """Run one declared experiment end to end and persist its artifacts.

    Writes under ``config.output_dir/<config.name>/``: the frozen splits and
    build manifest (``splits/``), the trained model (``model/``), per-split
    evaluation reports, the resolved config, and a ``result.json`` summary.

    Args:
        config: The complete experiment declaration.
        splits: Pre-built splits to reuse (the benchmark runner passes these
            so every model sees identical data); built from ``config.data``
            when omitted.

    Returns:
        The in-memory :class:`ExperimentResult`.

    Raises:
        DataError: if data loading/splitting fails or the test split is empty.
    """
    output_dir = config.output_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)

    if splits is None:
        splits = DatasetBuilder(config.data).build(
            config.split, target=config.target, output_dir=output_dir / "splits"
        )

    started = time.perf_counter()
    classifier = build_classifier(config)
    classifier.fit(splits.train)

    reports: dict[str, EvaluationReport] = {}
    predictions: dict[str, SplitPredictions] = {}
    for split_name, samples in (("validation", splits.validation), ("test", splits.test)):
        if not samples:
            _logger.info("split %r is empty; skipping evaluation", split_name)
            continue
        reports[split_name], predictions[split_name] = _evaluate_split(
            classifier, samples, split_name, config, calibration_bins=calibration_bins
        )
    if "test" not in reports:
        raise DataError(f"experiment {config.name!r} has no evaluable test split")
    seconds = time.perf_counter() - started

    model_path = classifier.save(output_dir / "model")
    save_experiment_config(config, output_dir / "config.yaml")
    for split_name, report in reports.items():
        report.save(output_dir / f"report_{split_name}.json")
    for split_name, split_predictions in predictions.items():
        split_predictions.save(output_dir / f"predictions_{split_name}.json")

    result = ExperimentResult(
        name=config.name,
        model=config.model.name,
        task=config.task.value,
        target=config.target.value,
        sizes=splits.sizes(),
        reports=reports,
        model_path=model_path,
        seconds=round(seconds, 3),
    )
    write_json(output_dir / "result.json", result.model_dump(mode="json"))
    _logger.info("experiment %r finished in %.1fs -> %s", config.name, seconds, output_dir)
    return result


def run_benchmark(
    config: ExperimentConfig,
    models: Sequence[ComponentConfig | str] | None = None,
    *,
    calibration_bins: int | None = None,
    n_jobs: int = 1,
) -> list[BenchmarkResult]:
    """Compare several models over one identical, frozen, speaker-disjoint split.

    Args:
        config: Base experiment declaration (data, features, split, target).
            Its ``model`` entry is the default competitor when ``models`` is
            omitted.
        models: The competitors (registry names or component configs). Each is
            trained and evaluated with the same features, samples, and split.
        calibration_bins: When given, each model's report gains a top-label
            calibration block (ECE/MCE/Brier) with this many uniform bins.
        n_jobs: Competitors to train in parallel. ``1`` runs in-process and
            sequentially (the default, and what the committed leaderboard uses).
            Above 1 (or ``-1`` for all cores) trains models in separate processes
            via joblib; because every fit re-seeds its own process, the results
            and their order are identical to the sequential run.

    Returns:
        One :class:`BenchmarkResult` per model, in the given order. Also
        writes ``benchmark.json`` and ``benchmark.md`` (sorted comparison
        table) under ``config.output_dir/<config.name>/``.
    """
    from joblib import Parallel, delayed

    entries = [
        entry if isinstance(entry, ComponentConfig) else ComponentConfig(name=str(entry))
        for entry in (models if models is not None else [config.model])
    ]
    output_dir = config.output_dir / config.name
    set_global_seed(config.seed)
    splits = DatasetBuilder(config.data).build(
        config.split, target=config.target, output_dir=output_dir / "splits"
    )

    # joblib preserves input order, and each model's fit re-seeds its own
    # (possibly separate) process, so the parallel results match the sequential
    # ones exactly. n_jobs=1 stays in-process, leaving the committed board's
    # byte-for-byte reproducibility untouched.
    results: list[BenchmarkResult] = Parallel(n_jobs=n_jobs)(
        delayed(_benchmark_one_model)(config, entry, splits, calibration_bins) for entry in entries
    )

    save_benchmark(results, output_dir / "benchmark.json")
    markdown = to_markdown_table(results)
    write_markdown(output_dir / "benchmark.md", markdown)
    _logger.info("benchmark %r: %d models compared -> %s", config.name, len(results), output_dir)
    return results


def _benchmark_one_model(
    config: ExperimentConfig,
    entry: ComponentConfig,
    splits: DatasetSplits,
    calibration_bins: int | None,
) -> BenchmarkResult:
    """Train and evaluate one competitor on the frozen split (a parallel unit).

    A module-level function so joblib's process backend can pickle it; its inputs
    and the returned :class:`BenchmarkResult` are all picklable.
    """
    candidate_config = config.model_copy(update={"model": entry})
    _logger.info("benchmark %r: training %s", config.name, entry.name)
    started = time.perf_counter()
    classifier = build_classifier(candidate_config)
    classifier.fit(splits.train)
    reports: dict[str, EvaluationReport] = {}
    predictions: dict[str, SplitPredictions] = {}
    for split_name, samples in splits.as_dict().items():
        if split_name == "train" or not samples:
            continue
        reports[split_name], predictions[split_name] = _evaluate_split(
            classifier, samples, split_name, candidate_config, calibration_bins=calibration_bins
        )
    return BenchmarkResult(
        experiment=config.name,
        model=entry.name,
        target_level=config.target.value,
        reports=reports,
        predictions=predictions,
        wall_seconds=round(time.perf_counter() - started, 3),
        n_train=len(splits.train),
        n_test=len(splits.test),
    )


#: TrainingConfig fields injected into a training-aware model whose metadata
#: does not narrow them. The neural fine-tuners accept all three; a model that
#: accepts only a subset (or none) declares its own ``training_knobs`` so this
#: function never has to know about it.
_DEFAULT_TRAINING_KNOBS = ("batch_size", "epochs", "learning_rate")


def build_classifier(config: ExperimentConfig) -> DialectClassifier:
    """Instantiate the classifier declared by an experiment config.

    For raw-input models (empty ``features``) registered as ``training_aware``
    (i.e. whose constructors accept the shared TrainingConfig knobs),
    ``config.training`` values are merged into the model params; explicit
    per-model params always win. A model declares *which* knobs its constructor
    accepts through its ``training_knobs`` metadata (default: all three); only
    those are injected, so a model that takes a subset (or renames them via
    factory aliases) can be training-aware without this function hardcoding the
    neural fine-tuners' knob set. The capability is declared at each model's
    registration site, so new models opt in without changes here.
    """
    from tulip.config.schemas import TrainingConfig
    from tulip.models import MODELS

    model = config.model
    metadata = MODELS.metadata(model.name)
    training_aware = metadata.get("training_aware", False)
    if not training_aware and config.training != TrainingConfig():
        _logger.warning(
            "experiment %r sets a training block, but model %r is not training-aware; "
            "the block is ignored; pass hyperparameters via model.params instead",
            config.name,
            model.name,
        )
    if not config.features and training_aware:
        knobs = metadata.get("training_knobs", _DEFAULT_TRAINING_KNOBS)
        training_values = {knob: getattr(config.training, knob) for knob in knobs}
        model = model.model_copy(update={"params": {**training_values, **model.params}})
    return DialectClassifier(
        model=model,
        features=config.features,
        task=config.task,
        target=config.target,
        abstain_threshold=config.abstain_threshold,
        seed=config.seed,
    )


__all__ = [
    "ExperimentResult",
    "build_classifier",
    "collect_predictions",
    "evaluate_samples",
    "run_benchmark",
    "run_experiment",
]
