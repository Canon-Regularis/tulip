"""Scoring a fitted classifier over a labelled split.

The one inference pass that both the aggregate
:class:`~tulip.evaluation.report.EvaluationReport` and the per-sample
:class:`~tulip.evaluation.predictions.SplitPredictions` derive from, plus the
public :func:`evaluate_samples` and :func:`collect_predictions` that wrap it.
Extracted from :mod:`tulip.pipeline.experiment` so the experiment/benchmark
*runners* (which build data, train, and persist) are separate from the scoring
they share: a caller that only wants to evaluate a model need not pull the
training-and-persistence machinery. The functions stay importable from their old
home, which re-exports them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tulip.core.exceptions import DataError
from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.predictions import PREDICTION_FLOAT_DIGITS, PredictionRecord, SplitPredictions
from tulip.evaluation.slicing import record_slice_keys
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.config.schemas import ExperimentConfig
    from tulip.core.types import Sample
    from tulip.evaluation.report import EvaluationReport
    from tulip.pipeline.classifier import DialectClassifier, LabelledBatch

_logger = get_logger(__name__)

__all__ = ["collect_predictions", "evaluate_samples"]


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
            **record_slice_keys(sample),
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
