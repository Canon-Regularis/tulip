"""Reproducible experiment and benchmark runners.

:func:`run_experiment` executes one fully declared experiment
(:class:`~tulip.config.schemas.ExperimentConfig`) end to end: seed, build
leakage-free splits, train, evaluate, and persist every artifact needed to
audit or reproduce the run. :func:`run_benchmark` trains several models over
the *identical* frozen split -- the reproducible-benchmark deliverable: there
is currently no widely adopted benchmark for Polish dialect identification,
and comparable numbers require byte-identical splits.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from tulip.config.loader import save_experiment_config
from tulip.config.schemas import ComponentConfig, ExperimentConfig
from tulip.core.exceptions import DataError
from tulip.core.types import Sample
from tulip.data.builder import DatasetBuilder
from tulip.data.splitting import DatasetSplits
from tulip.evaluation.benchmark import BenchmarkResult, save_benchmark, to_markdown_table
from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.report import EvaluationReport
from tulip.pipeline.classifier import DialectClassifier
from tulip.utils.io import write_json
from tulip.utils.logging import get_logger
from tulip.utils.seed import set_global_seed

_logger = get_logger(__name__)

#: Raw-input models whose constructors accept the shared TrainingConfig knobs.
_TRAINING_AWARE_MODELS = frozenset(
    {"herbert", "polish_roberta", "mbert", "xlm_roberta", "wav2vec2", "hubert", "whisper"}
)


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
    classifier = _build_classifier(config)
    classifier.fit(splits.train)

    reports: dict[str, EvaluationReport] = {}
    for split_name, samples in (("validation", splits.validation), ("test", splits.test)):
        if not samples:
            _logger.info("split %r is empty; skipping evaluation", split_name)
            continue
        reports[split_name] = _evaluate(classifier, samples, split_name, config)
    if "test" not in reports:
        raise DataError(f"experiment {config.name!r} has no evaluable test split")
    seconds = time.perf_counter() - started

    model_path = classifier.save(output_dir / "model")
    save_experiment_config(config, output_dir / "config.yaml")
    for split_name, report in reports.items():
        report.save(output_dir / f"report_{split_name}.json")

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
) -> list[BenchmarkResult]:
    """Compare several models over one identical, frozen, speaker-disjoint split.

    Args:
        config: Base experiment declaration (data, features, split, target).
            Its ``model`` entry is the default competitor when ``models`` is
            omitted.
        models: The competitors (registry names or component configs). Each is
            trained and evaluated with the same features, samples, and split.

    Returns:
        One :class:`BenchmarkResult` per model, in the given order. Also
        writes ``benchmark.json`` and ``benchmark.md`` (sorted comparison
        table) under ``config.output_dir/<config.name>/``.
    """
    entries = [
        entry if isinstance(entry, ComponentConfig) else ComponentConfig(name=str(entry))
        for entry in (models if models is not None else [config.model])
    ]
    output_dir = config.output_dir / config.name
    set_global_seed(config.seed)
    splits = DatasetBuilder(config.data).build(
        config.split, target=config.target, output_dir=output_dir / "splits"
    )

    results: list[BenchmarkResult] = []
    for entry in entries:
        model_config = config.model_copy(update={"model": entry})
        _logger.info("benchmark %r: training %s", config.name, entry.name)
        started = time.perf_counter()
        classifier = _build_classifier(model_config)
        classifier.fit(splits.train)
        reports = {
            split_name: _evaluate(classifier, samples, split_name, model_config)
            for split_name, samples in splits.as_dict().items()
            if split_name != "train" and samples
        }
        results.append(
            BenchmarkResult(
                experiment=config.name,
                model=entry.name,
                target_level=config.target.value,
                reports=reports,
                wall_seconds=round(time.perf_counter() - started, 3),
                n_train=len(splits.train),
                n_test=len(splits.test),
            )
        )

    save_benchmark(results, output_dir / "benchmark.json")
    markdown = to_markdown_table(results)
    (output_dir / "benchmark.md").write_text(markdown + "\n", encoding="utf-8", newline="\n")
    _logger.info("benchmark %r: %d models compared -> %s", config.name, len(results), output_dir)
    return results


def _build_classifier(config: ExperimentConfig) -> DialectClassifier:
    """Instantiate the classifier declared by an experiment config.

    For raw-input neural models (empty ``features``) whose wrappers accept the
    shared training knobs, ``config.training`` values are merged into the
    model params -- explicit per-model params always win.
    """
    model = config.model
    if not config.features and model.name.strip().lower() in _TRAINING_AWARE_MODELS:
        merged = {
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            **model.params,
        }
        model = model.model_copy(update={"params": merged})
    return DialectClassifier(
        model=model,
        features=config.features,
        task=config.task,
        target=config.target,
        abstain_threshold=config.abstain_threshold,
        seed=config.seed,
    )


def evaluate_samples(
    classifier: DialectClassifier,
    samples: Sequence[Sample],
    *,
    name: str = "eval",
    metadata: dict[str, str] | None = None,
) -> EvaluationReport:
    """Evaluate a fitted classifier on labelled samples.

    Abstention never applies here: evaluation scores the raw argmax so metrics
    stay comparable across abstention thresholds. Gold labels the model never
    saw are kept (they count as errors); when present they widen the label set
    beyond the probability columns, so ROC AUC is safely omitted by the
    metrics guard rather than misreported.

    Raises:
        DataError: if no sample carries both the input modality and a label
            at the classifier's target level.
    """
    raws, y_true, skipped = classifier._trainable(samples)
    if not raws:
        raise DataError(
            f"split {name!r} has no evaluable samples for target "
            f"{classifier.target.value!r} (skipped {skipped})"
        )
    if skipped:
        _logger.info("split %r: skipped %d unlabelled/modality-less samples", name, skipped)
    probabilities = classifier._predict_proba(raws)
    y_pred = [classifier.classes_[int(i)] for i in np.argmax(probabilities, axis=1)]
    labels = sorted(set(classifier.classes_) | set(y_true))
    return compute_metrics(
        y_true,
        y_pred,
        y_proba=probabilities,
        labels=labels,
        metadata={"split": name, "target": classifier.target.value, **(metadata or {})},
    )


def _evaluate(
    classifier: DialectClassifier,
    samples: Sequence[Sample],
    split_name: str,
    config: ExperimentConfig,
) -> EvaluationReport:
    """Evaluate one experiment split (config-aware metadata)."""
    return evaluate_samples(
        classifier,
        samples,
        name=split_name,
        metadata={"model": config.model.name, "experiment": config.name},
    )


__all__ = ["ExperimentResult", "evaluate_samples", "run_benchmark", "run_experiment"]
