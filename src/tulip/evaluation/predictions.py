"""Per-sample prediction records: the substrate for the rigor analyses.

The aggregate :class:`~tulip.evaluation.report.EvaluationReport` (a frozen core
contract) keeps only summary metrics plus the confusion matrix; it deliberately
does not retain per-sample predictions. Three benchmark-rigor analyses need
them, and share this one record type:

* paired significance testing (:mod:`tulip.evaluation.significance`) pairs each
  model's per-sample correctness on the identical frozen split;
* risk-coverage / selective prediction (:mod:`tulip.evaluation.selective`)
  sweeps a confidence threshold over the per-sample top probability;
* error analysis (:mod:`tulip.evaluation.error_analysis`) slices per-sample
  outcomes by source, speaker, length, and modality.

Records are *self-describing*: each carries its slice keys, so an analysis never
has to re-load the originating corpus (which may be unredistributable). Persisted
via :func:`tulip.evaluation._format.write_sorted_json` with probabilities rounded
to :data:`PREDICTION_FLOAT_DIGITS`, so a committed predictions dump regenerates
byte-for-byte.

This module imports nothing from :mod:`tulip.pipeline`: the collector that turns
a fitted classifier into a :class:`SplitPredictions` lives in
:func:`tulip.pipeline.experiment.collect_predictions` (pipeline depends on
evaluation, not the reverse), so there is no import cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tulip.core.exceptions import ConfigurationError
from tulip.evaluation._format import write_sorted_json
from tulip.utils.io import read_json

__all__ = [
    "PREDICTION_FLOAT_DIGITS",
    "PredictionRecord",
    "SplitPredictions",
]

#: Fixed rounding applied to every persisted probability, so a committed
#: predictions dump is byte-identical across re-runs even under trivial
#: floating-point noise (mirrors ``leaderboard.PROVENANCE_FLOAT_DIGITS``).
PREDICTION_FLOAT_DIGITS = 6


class PredictionRecord(BaseModel):
    """One evaluated sample: its gold label, prediction, and full distribution.

    ``proba`` is aligned to :attr:`SplitPredictions.labels`. The slice keys
    (``source``/``speaker_id``/``n_chars``/``modality``) are copied from the
    :class:`~tulip.core.types.Sample` so downstream slicing needs no corpus.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    y_true: str
    y_pred: str
    proba: tuple[float, ...]
    source: str = "unknown"
    speaker_id: str | None = None
    n_chars: int | None = Field(default=None, ge=0)
    modality: str = "text"

    @property
    def confidence(self) -> float:
        """Top predicted probability (0.0 when no distribution is present)."""
        return max(self.proba) if self.proba else 0.0

    @property
    def correct(self) -> bool:
        """Whether the raw-argmax prediction matched the gold label."""
        return self.y_true == self.y_pred


class SplitPredictions(BaseModel):
    """Every :class:`PredictionRecord` for one model on one evaluated split.

    Positionally comparable to another model's :class:`SplitPredictions` on the
    *same* split: the benchmark trains every competitor on the identical frozen,
    speaker-disjoint split, so both share the same surviving-sample order — which
    is what makes the significance tests genuinely paired.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    split: str
    labels: tuple[str, ...]
    records: tuple[PredictionRecord, ...]
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_alignment(self) -> SplitPredictions:
        """Reject empty/duplicate labels or a record whose ``proba`` misaligns."""
        n_labels = len(self.labels)
        if n_labels == 0:
            raise ValueError("SplitPredictions needs at least one label")
        if len(set(self.labels)) != n_labels:
            raise ValueError("labels must be unique")
        for record in self.records:
            if len(record.proba) != n_labels:
                raise ValueError(
                    f"record {record.id!r} has {len(record.proba)} probabilities, "
                    f"expected {n_labels} to align with labels"
                )
        return self

    def __len__(self) -> int:
        return len(self.records)

    # ---------------------------------------------------------------- arrays

    def true_labels(self) -> list[str]:
        """Gold labels, one per record, in record order."""
        return [record.y_true for record in self.records]

    def pred_labels(self) -> list[str]:
        """Raw-argmax predicted labels, one per record, in record order."""
        return [record.y_pred for record in self.records]

    def proba_matrix(self) -> np.ndarray:
        """The ``(n_samples, n_labels)`` probability matrix (columns == labels)."""
        if not self.records:
            return np.empty((0, len(self.labels)), dtype=np.float64)
        return np.asarray([record.proba for record in self.records], dtype=np.float64)

    def confidences(self) -> np.ndarray:
        """Per-record top probability, as a float array."""
        return np.asarray([record.confidence for record in self.records], dtype=np.float64)

    def correct(self) -> np.ndarray:
        """Per-record correctness (``y_true == y_pred``), as a boolean array."""
        return np.asarray([record.correct for record in self.records], dtype=bool)

    # ------------------------------------------------------------ persistence

    def save(self, path: Path | str) -> None:
        """Write to ``path`` as deterministic JSON (sorted keys, rounded proba)."""
        write_sorted_json(Path(path), self._payload())

    def _payload(self) -> dict[str, Any]:
        """The deterministic, JSON-native payload with probabilities rounded."""
        return {
            "labels": list(self.labels),
            "metadata": dict(self.metadata),
            "model": self.model,
            "records": [
                {
                    "id": record.id,
                    "modality": record.modality,
                    "n_chars": record.n_chars,
                    "proba": [round(value, PREDICTION_FLOAT_DIGITS) for value in record.proba],
                    "source": record.source,
                    "speaker_id": record.speaker_id,
                    "y_pred": record.y_pred,
                    "y_true": record.y_true,
                }
                for record in self.records
            ],
            "split": self.split,
        }

    @classmethod
    def load(cls, path: Path | str) -> SplitPredictions:
        """Read a predictions dump previously written by :meth:`save`.

        Raises:
            ConfigurationError: if the file is not a predictions dump.
        """
        data = read_json(Path(path))
        if not isinstance(data, dict) or "records" not in data or "labels" not in data:
            raise ConfigurationError(
                f"{path} is not a tulip predictions dump (expected 'labels' and 'records')"
            )
        return cls.model_validate(data)
