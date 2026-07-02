"""Compute a full :class:`~tulip.evaluation.report.EvaluationReport` from predictions.

Design choices worth knowing:

- ``zero_division=0`` everywhere, so classes that are never predicted (common
  with rare dialects) degrade metrics instead of raising warnings/errors.
- Macro one-vs-rest ROC AUC is *guarded*: it is reported only when honest to
  do so (probabilities supplied, columns aligned to ``labels``, every label
  present in ``y_true``); otherwise it is ``None`` with a debug log line.
- Balanced accuracy is derived from the per-class recalls of classes actually
  present in ``y_true`` (identical to scikit-learn's definition) so that a
  ``labels`` superset never distorts it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation.report import ClassMetrics, EvaluationReport
from tulip.utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: Any | None = None,
    labels: Sequence[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Evaluate predictions against gold labels for binary or multiclass tasks.

    Args:
        y_true: Gold labels, one per sample. Values are coerced to ``str``.
        y_pred: Predicted labels, aligned with ``y_true``.
        y_proba: Optional probability matrix of shape ``(n_samples, n_labels)``
            with columns ordered like ``labels``; enables macro one-vs-rest
            ROC AUC when usable.
        labels: Explicit class order for the report and confusion matrix.
            Defaults to the sorted union of ``y_true`` and ``y_pred``. Every
            observed label must be included when this is given.
        metadata: Free-form context stored on the report (e.g. model name,
            target level, split name).

    Returns:
        A frozen :class:`EvaluationReport` with overall, per-class, and
        confusion-matrix results.

    Raises:
        ConfigurationError: If ``y_true``/``y_pred`` lengths differ, the inputs
            are empty, ``y_proba`` has a mismatched number of rows, ``labels``
            contains duplicates, or an observed label is missing from an
            explicit ``labels``.
    """
    true_list = [str(value) for value in y_true]
    pred_list = [str(value) for value in y_pred]
    if len(true_list) != len(pred_list):
        raise ConfigurationError(
            f"y_true and y_pred must have the same length; got {len(true_list)} true labels "
            f"and {len(pred_list)} predictions"
        )
    if not true_list:
        raise ConfigurationError("cannot compute metrics on zero samples")

    label_list = _resolve_labels(true_list, pred_list, labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_list, pred_list, labels=label_list, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_list, pred_list, labels=label_list, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_list, pred_list, labels=label_list, average="weighted", zero_division=0
    )

    # Balanced accuracy == mean recall over classes present in y_true.
    present_recalls = [float(r) for r, s in zip(recall, support, strict=True) if s > 0]
    balanced_accuracy = float(np.mean(present_recalls)) if present_recalls else 0.0

    per_class = {
        label: ClassMetrics(precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for label, p, r, f, s in zip(label_list, precision, recall, f1, support, strict=True)
    }
    confusion = tuple(
        tuple(int(cell) for cell in row)
        for row in confusion_matrix(true_list, pred_list, labels=label_list)
    )

    return EvaluationReport(
        accuracy=float(accuracy_score(true_list, pred_list)),
        balanced_accuracy=balanced_accuracy,
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
        roc_auc_macro_ovr=_guarded_roc_auc(true_list, y_proba, label_list),
        labels=tuple(label_list),
        per_class=per_class,
        confusion=confusion,
        n_samples=len(true_list),
        metadata=dict(metadata) if metadata else {},
    )


def _resolve_labels(
    true_list: list[str], pred_list: list[str], labels: Sequence[Any] | None
) -> list[str]:
    """Return the class order, defaulting to the sorted union of observed labels."""
    observed = set(true_list) | set(pred_list)
    if labels is None:
        return sorted(observed)
    label_list = [str(value) for value in labels]
    if len(set(label_list)) != len(label_list):
        raise ConfigurationError(f"labels contains duplicates: {label_list!r}")
    unknown = observed - set(label_list)
    if unknown:
        raise ConfigurationError(
            f"labels observed in y_true/y_pred are missing from `labels`: {sorted(unknown)}"
        )
    return label_list


def _guarded_roc_auc(
    true_list: list[str], y_proba: Any | None, label_list: list[str]
) -> float | None:
    """Macro one-vs-rest ROC AUC, or ``None`` (with a debug log) when not computable.

    Only a row-count mismatch raises (it means predictions and probabilities
    describe different samples — a caller bug); every other shortfall simply
    disables the metric.
    """
    if y_proba is None:
        logger.debug("ROC AUC skipped: y_proba not provided")
        return None
    try:
        proba = np.asarray(y_proba, dtype=float)
    except (TypeError, ValueError):
        logger.debug("ROC AUC skipped: y_proba is not a numeric array")
        return None
    if proba.ndim != 2:
        logger.debug("ROC AUC skipped: y_proba must be 2-D, got ndim=%d", proba.ndim)
        return None
    if proba.shape[0] != len(true_list):
        raise ConfigurationError(
            f"y_proba has {proba.shape[0]} rows but there are {len(true_list)} samples"
        )
    if proba.shape[1] != len(label_list):
        logger.debug(
            "ROC AUC skipped: y_proba has %d columns but there are %d labels",
            proba.shape[1],
            len(label_list),
        )
        return None
    missing = sorted(set(label_list) - set(true_list))
    if missing:
        logger.debug("ROC AUC skipped: labels absent from y_true: %s", missing)
        return None
    try:
        if len(label_list) == 2:
            # Binary macro-OVR AUC equals the plain binary AUC of the second class.
            y_binary = [1 if value == label_list[1] else 0 for value in true_list]
            return float(roc_auc_score(y_binary, proba[:, 1]))
        return float(
            roc_auc_score(true_list, proba, multi_class="ovr", average="macro", labels=label_list)
        )
    except ValueError as exc:  # e.g. rows not summing to 1 for multiclass scoring
        logger.debug("ROC AUC skipped: %s", exc)
        return None
