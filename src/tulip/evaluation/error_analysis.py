"""Error analysis and fairness/robustness slices over per-sample predictions.

A single macro-F1 says *how much* a model is wrong, never *where*. This module
derives the diagnosis a confusion matrix alone cannot give:

* **Top confused pairs**: which dialects are systematically merged (the
  off-diagonal, ranked), so a linguist can see *podhale <-> spisz* rather than a
  grid of counts.
* **Hard exemplars**: the highest-confidence mistakes (the model was sure and
  wrong) and the least-confident calls, by sample id, so they can be pulled and
  inspected.
* **Slice metrics**: accuracy and macro-F1 recomputed per source corpus,
  speaker, modality, and length band, exposing the fairness/leakage skew a
  pooled number hides. Low-support slices are flagged, never reported as
  headline.

It consumes only :class:`~tulip.evaluation.predictions.SplitPredictions`, whose
records already carry the slice keys, so it needs no access to the (possibly
unredistributable) corpus. An optional ``texts`` map keyed by sample id enriches
the exemplars with a snippet when the corpus *is* available. Every ordering is
total and explicit, so a committed error report regenerates byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tulip.evaluation._format import format_metric, markdown_table, write_sorted_json
from tulip.evaluation.metrics import compute_metrics

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tulip.evaluation.predictions import PredictionRecord, SplitPredictions

__all__ = [
    "ConfusedPair",
    "ErrorReport",
    "Exemplar",
    "SliceMetric",
    "error_report",
    "slice_metrics",
    "top_confused_pairs",
]

#: Slices with fewer than this many samples are flagged low-support: their
#: metrics are noisy and must not be read as headline fairness findings.
DEFAULT_LOW_SUPPORT = 5

#: Inclusive upper edges (in characters) of the length bands, plus an open top.
_LENGTH_BANDS: tuple[tuple[str, int | None], ...] = (
    ("<=40", 40),
    ("41-80", 80),
    ("81-160", 160),
    (">160", None),
)

#: Longest snippet shown for an exemplar (characters), when texts are supplied.
_SNIPPET_CHARS = 80


class ConfusedPair(BaseModel):
    """A (true -> predicted) confusion with its count, off the diagonal."""

    model_config = ConfigDict(frozen=True)

    true_label: str
    pred_label: str
    count: int = Field(ge=1)


class Exemplar(BaseModel):
    """A single sample singled out for inspection."""

    model_config = ConfigDict(frozen=True)

    id: str
    y_true: str
    y_pred: str
    confidence: float = Field(ge=0.0, le=1.0)
    text: str | None = None


class SliceMetric(BaseModel):
    """Metrics for one data slice (e.g. source == 'bigos')."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    value: str
    n: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)
    low_support: bool


class ErrorReport(BaseModel):
    """Derived error analysis for one model on one split."""

    model_config = ConfigDict(frozen=True)

    model: str
    split: str
    n_samples: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    confused_pairs: tuple[ConfusedPair, ...]
    hardest_errors: tuple[Exemplar, ...]
    least_confident: tuple[Exemplar, ...]
    slices: tuple[SliceMetric, ...]

    def to_markdown(self) -> str:
        """Render the report as markdown sections."""
        parts = [
            f"# Error analysis: {self.model} ({self.split})",
            markdown_table(
                ("Metric", "Value"),
                [("Samples", str(self.n_samples)), ("Accuracy", format_metric(self.accuracy))],
            ),
            "## Most-confused pairs",
            markdown_table(
                ("True", "Predicted", "Count"),
                [(p.true_label, p.pred_label, str(p.count)) for p in self.confused_pairs]
                or [("n/a", "n/a", "0")],
            ),
            "## Highest-confidence errors",
            markdown_table(
                ("Sample", "True", "Predicted", "Confidence"),
                [
                    (e.id, e.y_true, e.y_pred, format_metric(e.confidence))
                    for e in self.hardest_errors
                ]
                or [("n/a", "n/a", "n/a", "n/a")],
            ),
            "## Slice metrics (low-support marked *)",
            markdown_table(
                ("Dimension", "Value", "N", "Accuracy", "F1 (macro)"),
                [
                    (
                        s.dimension,
                        s.value + (" *" if s.low_support else ""),
                        str(s.n),
                        format_metric(s.accuracy),
                        format_metric(s.f1_macro),
                    )
                    for s in self.slices
                ],
            ),
        ]
        return "\n\n".join(parts)

    def save(self, path: Path | str) -> None:
        """Write the report as deterministic JSON (sorted keys)."""
        write_sorted_json(Path(path), self.model_dump(mode="json"))


def top_confused_pairs(predictions: SplitPredictions, *, top_k: int = 10) -> list[ConfusedPair]:
    """Rank the (true -> predicted) confusions, most frequent first.

    Only genuine confusions are returned (correct predictions are on the
    diagonal and excluded). Ties are broken by ``(true_label, pred_label)`` for a
    total, reproducible order.
    """
    counts: dict[tuple[str, str], int] = {}
    for record in predictions.records:
        if record.y_true != record.y_pred:
            key = (record.y_true, record.y_pred)
            counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        ConfusedPair(true_label=true, pred_label=pred, count=count)
        for (true, pred), count in ranked[:top_k]
    ]


def slice_metrics(
    predictions: SplitPredictions, *, low_support: int = DEFAULT_LOW_SUPPORT
) -> list[SliceMetric]:
    """Recompute accuracy and macro-F1 for each source/speaker/modality/length slice.

    Slices are produced for every dimension whose key is present, sorted by
    ``(dimension, value)``. A slice with fewer than ``low_support`` samples is
    flagged rather than dropped, so a fairness gap on a rare group is still
    visible but never mistaken for a headline result.
    """
    dimensions: dict[str, list[tuple[str, PredictionRecord]]] = {
        "source": [],
        "speaker_id": [],
        "modality": [],
        "length": [],
    }
    for record in predictions.records:
        dimensions["source"].append((record.source, record))
        if record.speaker_id is not None:
            dimensions["speaker_id"].append((record.speaker_id, record))
        dimensions["modality"].append((record.modality, record))
        if record.n_chars is not None:
            dimensions["length"].append((_length_band(record.n_chars), record))

    metrics: list[SliceMetric] = []
    for dimension, tagged in dimensions.items():
        grouped: dict[str, list[PredictionRecord]] = {}
        for value, record in tagged:
            grouped.setdefault(value, []).append(record)
        for value in sorted(grouped):
            records = grouped[value]
            report = compute_metrics([r.y_true for r in records], [r.y_pred for r in records])
            metrics.append(
                SliceMetric(
                    dimension=dimension,
                    value=value,
                    n=len(records),
                    accuracy=report.accuracy,
                    f1_macro=report.f1_macro,
                    low_support=len(records) < low_support,
                )
            )
    return metrics


def error_report(
    predictions: SplitPredictions,
    *,
    texts: Mapping[str, str] | None = None,
    top_k: int = 10,
    low_support: int = DEFAULT_LOW_SUPPORT,
) -> ErrorReport:
    """Assemble a full :class:`ErrorReport` from per-sample predictions.

    Args:
        predictions: The per-sample records for one model on one split.
        texts: Optional ``{sample_id: text}`` map; when given, exemplars carry a
            truncated snippet. Omit it to keep the report corpus-free.
        top_k: How many confused pairs and exemplars of each kind to keep.
        low_support: Slice size below which a slice is flagged low-support.

    Returns:
        A frozen :class:`ErrorReport`.

    Raises:
        ConfigurationError: if there are no predictions (via ``compute_metrics``).
    """
    accuracy = compute_metrics(predictions.true_labels(), predictions.pred_labels()).accuracy
    errors = [record for record in predictions.records if not record.correct]
    hardest = sorted(errors, key=lambda r: (-r.confidence, r.id))[:top_k]
    least_confident = sorted(predictions.records, key=lambda r: (r.confidence, r.id))[:top_k]
    return ErrorReport(
        model=predictions.model,
        split=predictions.split,
        n_samples=len(predictions),
        accuracy=accuracy,
        confused_pairs=tuple(top_confused_pairs(predictions, top_k=top_k)),
        hardest_errors=tuple(_exemplar(record, texts) for record in hardest),
        least_confident=tuple(_exemplar(record, texts) for record in least_confident),
        slices=tuple(slice_metrics(predictions, low_support=low_support)),
    )


def _exemplar(record: PredictionRecord, texts: Mapping[str, str] | None) -> Exemplar:
    """Build an :class:`Exemplar`, attaching a truncated snippet when available."""
    snippet = None
    if texts is not None and record.id in texts:
        text = texts[record.id]
        snippet = text[:_SNIPPET_CHARS] + ("..." if len(text) > _SNIPPET_CHARS else "")
    return Exemplar(
        id=record.id,
        y_true=record.y_true,
        y_pred=record.y_pred,
        confidence=record.confidence,
        text=snippet,
    )


def _length_band(n_chars: int) -> str:
    """Bucket a character count into a fixed, ordered length band."""
    for label, upper in _LENGTH_BANDS:
        if upper is None or n_chars <= upper:
            return label
    return _LENGTH_BANDS[-1][0]  # pragma: no cover - the open top always matches
