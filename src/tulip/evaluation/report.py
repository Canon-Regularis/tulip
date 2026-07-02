"""The :class:`EvaluationReport` record: one classifier evaluated on one split.

The report is a frozen pydantic model so results can be persisted, compared,
and embedded in benchmark tables without fear of accidental mutation. It is
produced by :func:`tulip.evaluation.metrics.compute_metrics` and consumed by
the confusion-matrix helpers, the benchmark comparison layer, and the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tulip.evaluation._format import format_metric, markdown_table
from tulip.utils.io import read_json, write_json


class ClassMetrics(BaseModel):
    """Precision/recall/F1 for a single class, with its support in ``y_true``."""

    model_config = ConfigDict(frozen=True)

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=0)


class EvaluationReport(BaseModel):
    """All standard classification metrics for one model on one dataset split.

    ``labels`` fixes the class order; ``confusion`` and ``per_class`` are
    aligned to it (``confusion[i][j]`` counts samples whose true label is
    ``labels[i]`` and predicted label is ``labels[j]``).

    ``roc_auc_macro_ovr`` is ``None`` when it cannot be computed honestly
    (no probability estimates, or a class absent from the true labels) rather
    than silently misleading.
    """

    model_config = ConfigDict(frozen=True)

    accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    precision_macro: float = Field(ge=0.0, le=1.0)
    recall_macro: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)
    precision_weighted: float = Field(ge=0.0, le=1.0)
    recall_weighted: float = Field(ge=0.0, le=1.0)
    f1_weighted: float = Field(ge=0.0, le=1.0)
    roc_auc_macro_ovr: float | None = Field(default=None, ge=0.0, le=1.0)
    labels: tuple[str, ...]
    per_class: dict[str, ClassMetrics]
    confusion: tuple[tuple[int, ...], ...]
    n_samples: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_alignment(self) -> EvaluationReport:
        """Reject reports whose per-class data is not aligned to ``labels``."""
        n = len(self.labels)
        if n == 0:
            raise ValueError("an evaluation report needs at least one label")
        if len(set(self.labels)) != n:
            raise ValueError("labels must be unique")
        if set(self.per_class) != set(self.labels):
            raise ValueError("per_class keys must match labels exactly")
        if len(self.confusion) != n or any(len(row) != n for row in self.confusion):
            raise ValueError(f"confusion matrix must be {n}x{n} to align with labels")
        return self

    def summary_line(self) -> str:
        """Return a compact single-line summary suitable for logs.

        Returns:
            A line such as ``"nb/test: n=40 acc=0.8500 bacc=0.8421 f1_macro=0.8397
            f1_weighted=0.8478 auc=0.9450"``; the ``model/split`` prefix appears
            only when present in ``metadata``.
        """
        prefix_bits = [str(self.metadata[k]) for k in ("model", "split") if self.metadata.get(k)]
        prefix = "/".join(prefix_bits) + ": " if prefix_bits else ""
        return (
            f"{prefix}n={self.n_samples}"
            f" acc={format_metric(self.accuracy)}"
            f" bacc={format_metric(self.balanced_accuracy)}"
            f" f1_macro={format_metric(self.f1_macro)}"
            f" f1_weighted={format_metric(self.f1_weighted)}"
            f" auc={format_metric(self.roc_auc_macro_ovr)}"
        )

    def to_markdown(self) -> str:
        """Render the report as markdown: overall summary plus a per-class table.

        Returns:
            A markdown document (heading, metadata bullets, overall-metrics
            table, per-class table) ending without a trailing newline.
        """
        title = "Evaluation report"
        if self.metadata.get("model"):
            title += f" — {self.metadata['model']}"
        if self.metadata.get("split"):
            title += f" ({self.metadata['split']})"
        parts = [f"# {title}"]

        extra = {k: v for k, v in sorted(self.metadata.items()) if k not in ("model", "split")}
        if extra:
            parts.append("\n".join(f"- {key}: {value}" for key, value in extra.items()))

        overall_rows = [
            ("Samples", str(self.n_samples)),
            ("Classes", str(len(self.labels))),
            ("Accuracy", format_metric(self.accuracy)),
            ("Balanced accuracy", format_metric(self.balanced_accuracy)),
            ("Precision (macro)", format_metric(self.precision_macro)),
            ("Recall (macro)", format_metric(self.recall_macro)),
            ("F1 (macro)", format_metric(self.f1_macro)),
            ("Precision (weighted)", format_metric(self.precision_weighted)),
            ("Recall (weighted)", format_metric(self.recall_weighted)),
            ("F1 (weighted)", format_metric(self.f1_weighted)),
            ("ROC AUC (macro OVR)", format_metric(self.roc_auc_macro_ovr)),
        ]
        parts.append(markdown_table(("Metric", "Value"), overall_rows))

        per_class_rows = [
            (
                label,
                format_metric(self.per_class[label].precision),
                format_metric(self.per_class[label].recall),
                format_metric(self.per_class[label].f1),
                str(self.per_class[label].support),
            )
            for label in self.labels
        ]
        parts.append("## Per-class metrics")
        parts.append(
            markdown_table(("Class", "Precision", "Recall", "F1", "Support"), per_class_rows)
        )
        return "\n\n".join(parts)

    def save(self, path: Path | str) -> None:
        """Write the report to ``path`` as UTF-8 JSON (parents created)."""
        write_json(Path(path), self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: Path | str) -> EvaluationReport:
        """Read a report previously written by :meth:`save`.

        Args:
            path: JSON file produced by :meth:`save`.

        Returns:
            The reconstructed report; equal (``==``) to the saved instance.
        """
        return cls.model_validate(read_json(Path(path)))
