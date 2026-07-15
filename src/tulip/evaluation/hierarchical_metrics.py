"""Hierarchical metrics that credit partial family-to-dialect correctness.

A flat metric scores a prediction as fully right or fully wrong. That throws
away structure the dialect taxonomy carries: predicting Spisz for Podhale (both
Lesser Polish) is a better error than predicting Kashubian, yet accuracy treats
them the same. These metrics use the family tree to give partial credit.

Four numbers, from strict to lenient:

* exact accuracy: the flat accuracy, full credit only for the right dialect;
* family accuracy: credit when the predicted dialect shares the gold family;
* partial credit: full credit for an exact hit, :data:`PARTIAL_CREDIT_WEIGHT`
  for a family-only hit, zero otherwise;
* hierarchical F1: micro-F1 over each label augmented with its family, the
  standard hierarchical measure.

The family of each label comes from :func:`tulip.labels.taxonomy.family_for`.
A label outside the taxonomy (a corpus-specific string, or a label that is
already a family) has no known parent and scores credit only on an exact match,
never a crash. Every function is pure, so a report is deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import ConfigurationError
from tulip.labels.taxonomy import LabelLevel, family_for

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["PARTIAL_CREDIT_WEIGHT", "HierarchicalReport", "hierarchical_metrics"]

#: Credit for a family-correct but dialect-wrong prediction. Fixed and stated so
#: the partial-credit score is not an arbitrary knob.
PARTIAL_CREDIT_WEIGHT = 0.5


class HierarchicalReport(BaseModel):
    """Flat and hierarchical accuracy, plus the hierarchical F1."""

    model_config = ConfigDict(frozen=True)

    level: str
    n_samples: int = Field(ge=1)
    partial_credit_weight: float = Field(ge=0.0, le=1.0)
    exact_accuracy: float = Field(ge=0.0, le=1.0)
    family_accuracy: float = Field(ge=0.0, le=1.0)
    partial_credit: float = Field(ge=0.0, le=1.0)
    hierarchical_f1: float = Field(ge=0.0, le=1.0)

    def to_markdown(self) -> str:
        """Render the metrics as a small markdown table."""
        from tulip.evaluation._format import format_metric, markdown_table

        rows = [
            ("exact accuracy", format_metric(self.exact_accuracy)),
            ("family accuracy", format_metric(self.family_accuracy)),
            (
                f"partial credit (family={self.partial_credit_weight})",
                format_metric(self.partial_credit),
            ),
            ("hierarchical F1", format_metric(self.hierarchical_f1)),
        ]
        title = f"# Hierarchical metrics - {self.level} ({self.n_samples} samples)"
        return f"{title}\n\n{markdown_table(('Metric', 'Value'), rows)}"


def _family_key(label: str) -> str:
    """The label's family value, or the label itself when it has no known family."""
    family = family_for(label)
    return family.value if family is not None else label


def _ancestors(label: str) -> set[str]:
    """The label augmented with its family (just the label if it has no family)."""
    family = family_for(label)
    return {family.value, label} if family is not None else {label}


def hierarchical_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    level: LabelLevel = LabelLevel.DIALECT,
) -> HierarchicalReport:
    """Compute exact, family, partial-credit accuracy and hierarchical F1.

    Args:
        y_true: Gold labels, one per sample.
        y_pred: Predicted labels, aligned with ``y_true``.
        level: Granularity of the labels, recorded on the report.

    Returns:
        A :class:`HierarchicalReport`.

    Raises:
        ConfigurationError: if the inputs differ in length or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ConfigurationError(
            f"y_true and y_pred differ in length: {len(y_true)} vs {len(y_pred)}"
        )
    if not y_true:
        raise ConfigurationError("hierarchical_metrics needs at least one sample")

    n = len(y_true)
    exact = family = 0
    credit = 0.0
    overlap = pred_size = true_size = 0
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        same_family = _family_key(true_label) == _family_key(pred_label)
        if true_label == pred_label:
            exact += 1
            credit += 1.0
        elif same_family:
            credit += PARTIAL_CREDIT_WEIGHT
        if same_family:
            family += 1
        true_ancestors = _ancestors(true_label)
        pred_ancestors = _ancestors(pred_label)
        overlap += len(true_ancestors & pred_ancestors)
        pred_size += len(pred_ancestors)
        true_size += len(true_ancestors)

    precision = overlap / pred_size if pred_size else 0.0
    recall = overlap / true_size if true_size else 0.0
    hierarchical_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return HierarchicalReport(
        level=level.value,
        n_samples=n,
        partial_credit_weight=PARTIAL_CREDIT_WEIGHT,
        exact_accuracy=exact / n,
        family_accuracy=family / n,
        partial_credit=credit / n,
        hierarchical_f1=hierarchical_f1,
    )
