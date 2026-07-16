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

import numpy as np
from pydantic import BaseModel, ConfigDict

from tulip._serialize import write_markdown
from tulip.config.loader import save_experiment_config
from tulip.config.schemas import ComponentConfig, ExperimentConfig
from tulip.core.exceptions import DataError
from tulip.data.builder import DatasetBuilder
from tulip.evaluation.benchmark import BenchmarkResult, save_benchmark, to_markdown_table
from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.predictions import PREDICTION_FLOAT_DIGITS, PredictionRecord, SplitPredictions
from tulip.evaluation.report import EvaluationReport
from tulip.pipeline.classifier import DialectClassifier, LabelledBatch
from tulip.utils.io import write_json
from tulip.utils.logging import get_logger
from tulip.utils.seed import set_global_seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.core.types import Sample
    from tulip.data.splitting import DatasetSplits

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
        results.append(
            BenchmarkResult(
                experiment=config.name,
                model=entry.name,
                target_level=config.target.value,
                reports=reports,
                predictions=predictions,
                wall_seconds=round(time.perf_counter() - started, 3),
                n_train=len(splits.train),
                n_test=len(splits.test),
            )
        )

    save_benchmark(results, output_dir / "benchmark.json")
    markdown = to_markdown_table(results)
    write_markdown(output_dir / "benchmark.md", markdown)
    _logger.info("benchmark %r: %d models compared -> %s", config.name, len(results), output_dir)
    return results


def build_classifier(config: ExperimentConfig) -> DialectClassifier:
    """Instantiate the classifier declared by an experiment config.

    For raw-input models (empty ``features``) registered as ``training_aware``
    (i.e. whose constructors accept the shared TrainingConfig knobs),
    ``config.training`` values are merged into the model params; explicit
    per-model params always win. The capability is declared at each model's
    registration site (see :meth:`tulip.core.registry.Registry.add`), so new
    models opt in without changes here.
    """
    from tulip.config.schemas import TrainingConfig
    from tulip.models import MODELS

    model = config.model
    training_aware = MODELS.metadata(model.name).get("training_aware", False)
    if not training_aware and config.training != TrainingConfig():
        _logger.warning(
            "experiment %r sets a training block, but model %r is not training-aware; "
            "the block is ignored; pass hyperparameters via model.params instead",
            config.name,
            model.name,
        )
    if not config.features and training_aware:
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
    calibration_bins: int | None = None,
) -> EvaluationReport:
    """Evaluate a fitted classifier on labelled samples.

    Abstention never applies here: evaluation scores the raw argmax so metrics
    stay comparable across abstention thresholds. Gold labels the model never
    saw are kept (they count as errors); when present they widen the label set
    beyond the probability columns, so ROC AUC is safely omitted by the
    metrics guard rather than misreported.

    Args:
        classifier: A fitted classifier.
        samples: Labelled samples to score.
        name: Split name recorded in the report metadata.
        metadata: Extra free-form report metadata.
        calibration_bins: When given, also populate the report's calibration
            block (top-label ECE/MCE and Brier) with this many uniform bins.
            Defaults to ``None`` (no calibration), so existing artifacts are
            unchanged until it is explicitly requested.

    Raises:
        DataError: if no sample carries both the input modality and a label
            at the classifier's target level.
    """
    batch, probabilities, y_pred = _score_split(classifier, samples, name)
    return _report_from(classifier, batch, probabilities, y_pred, name, metadata, calibration_bins)


def collect_predictions(
    classifier: DialectClassifier,
    samples: Sequence[Sample],
    *,
    name: str = "eval",
    metadata: dict[str, str] | None = None,
) -> SplitPredictions:
    """Evaluate a fitted classifier and keep the per-sample predictions.

    Scores the same raw argmax as :func:`evaluate_samples` (no abstention), so
    the returned records align exactly with that split's
    :class:`~tulip.evaluation.report.EvaluationReport`. Each record also carries
    self-describing slice keys (source, speaker, length, modality) so the
    downstream significance / selective-prediction / error-analysis layers need
    never re-load the originating corpus.

    Raises:
        DataError: if no sample carries both the input modality and a label
            at the classifier's target level.
    """
    batch, probabilities, y_pred = _score_split(classifier, samples, name)
    return _predictions_from(classifier, batch, probabilities, y_pred, name, metadata)


def _score_split(
    classifier: DialectClassifier, samples: Sequence[Sample], name: str
) -> tuple[LabelledBatch, np.ndarray, list[str]]:
    """Run inference once for a split: filtered batch, probabilities, argmax labels.

    The single shared inference pass behind both the aggregate report and the
    per-sample predictions, so a runner that wants both pays for prediction only
    once (it matters for the neural models).

    Raises:
        DataError: if no sample carries both the input modality and a label.
    """
    batch = classifier.labelled_batch(samples)
    if not batch.raws:
        raise DataError(
            f"split {name!r} has no evaluable samples for target "
            f"{classifier.target.value!r} (skipped {batch.n_skipped})"
        )
    if batch.n_skipped:
        _logger.info("split %r: skipped %d unlabelled/modality-less samples", name, batch.n_skipped)
    probabilities = classifier.predict_proba(batch.raws)
    y_pred = [classifier.classes_[int(i)] for i in np.argmax(probabilities, axis=1)]
    return batch, probabilities, y_pred


def _report_from(
    classifier: DialectClassifier,
    batch: LabelledBatch,
    probabilities: np.ndarray,
    y_pred: Sequence[str],
    name: str,
    metadata: dict[str, str] | None,
    calibration_bins: int | None = None,
) -> EvaluationReport:
    """Build the aggregate report from an already-scored split."""
    labels = sorted(set(classifier.classes_) | set(batch.labels))
    return compute_metrics(
        batch.labels,
        y_pred,
        y_proba=probabilities,
        labels=labels,
        metadata={"split": name, "target": classifier.target.value, **(metadata or {})},
        calibration_bins=calibration_bins,
    )


def _predictions_from(
    classifier: DialectClassifier,
    batch: LabelledBatch,
    probabilities: np.ndarray,
    y_pred: Sequence[str],
    name: str,
    metadata: dict[str, str] | None,
) -> SplitPredictions:
    """Build per-sample records from an already-scored split."""
    records = tuple(
        PredictionRecord(
            id=sample.id,
            y_true=true_label,
            y_pred=pred_label,
            proba=tuple(round(float(value), PREDICTION_FLOAT_DIGITS) for value in row),
            source=sample.source,
            speaker_id=sample.speaker_id,
            n_chars=len(sample.text) if sample.text is not None else None,
            modality=classifier.task.value,
        )
        for sample, true_label, pred_label, row in zip(
            batch.samples, batch.labels, y_pred, probabilities, strict=True
        )
    )
    return SplitPredictions(
        model=classifier.model_config.name,
        split=name,
        labels=classifier.classes_,
        records=records,
        metadata={"target": classifier.target.value, **(metadata or {})},
    )


def _evaluate_split(
    classifier: DialectClassifier,
    samples: Sequence[Sample],
    split_name: str,
    config: ExperimentConfig,
    *,
    calibration_bins: int | None = None,
) -> tuple[EvaluationReport, SplitPredictions]:
    """Evaluate one experiment split, returning report and predictions together.

    Scores the split once (:func:`_score_split`) and derives both artifacts, so
    the runners never pay for inference twice. Metadata is config-aware.
    """
    metadata = {"model": config.model.name, "experiment": config.name}
    batch, probabilities, y_pred = _score_split(classifier, samples, split_name)
    return (
        _report_from(
            classifier, batch, probabilities, y_pred, split_name, metadata, calibration_bins
        ),
        _predictions_from(classifier, batch, probabilities, y_pred, split_name, metadata),
    )


__all__ = [
    "ExperimentResult",
    "build_classifier",
    "collect_predictions",
    "evaluate_samples",
    "run_benchmark",
    "run_experiment",
]
