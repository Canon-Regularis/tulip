"""Efficiency metrics: inference latency, model size, and parameter count.

A benchmark that ranks only by accuracy hides the cost of a win. A transformer
that beats a logistic-regression baseline by a point of F1 may be a hundred times
larger and slower, which matters for a model anyone actually deploys. This module
measures the three costs a reader compares against accuracy: median per-sample
inference latency, the size of the saved model on disk, and the parameter count.

These numbers are machine dependent, so they are kept strictly off the byte-stable
artifacts. An :class:`EfficiencyRecord` is written only to its own
``efficiency.json``, which, like ``leaderboard.json``, is documented as outside
the reproducibility guarantee: two machines will time the same model differently,
and that is expected. The measurement is deliberately decoupled from the
leaderboard run, so nothing machine dependent can leak into ``leaderboard.md`` or
``provenance.json``.

Parameter counting is best effort: it reads a linear model's coefficients or a
torch module's tensors and returns ``None`` for a model whose parameter count has
no single honest definition (a random forest, a naive Bayes table), rather than
inventing one.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import write_sorted_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.pipeline.classifier import DialectClassifier

__all__ = [
    "EFFICIENCY_JSON",
    "EfficiencyRecord",
    "count_parameters",
    "measure_efficiency",
    "model_size_bytes",
    "write_efficiency",
]

#: Artifact name for the efficiency dump. Machine dependent, so excluded from the
#: byte-identical reproducibility guarantee, exactly like ``leaderboard.json``.
EFFICIENCY_JSON = "efficiency.json"

#: Digits kept on the latency reading; it varies per machine regardless, so this
#: is for readability, not reproducibility.
_LATENCY_DIGITS = 4


class EfficiencyRecord(BaseModel):
    """The measured cost of one model, alongside its accuracy on the board.

    Every field here is machine dependent (or, for size, build dependent), so a
    record is never part of a byte-stable artifact.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    experiment: str | None = None
    n_samples: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    model_size_bytes: int | None = Field(default=None, ge=0)
    n_params: int | None = Field(default=None, ge=0)


def measure_efficiency(
    classifier: DialectClassifier,
    samples: Sequence[Any],
    *,
    model: str,
    experiment: str | None = None,
    repeats: int = 3,
    model_dir: Path | str | None = None,
) -> EfficiencyRecord:
    """Measure one model's inference latency, size, and parameter count.

    Args:
        classifier: A fitted classifier exposing ``predict_samples``.
        samples: The batch to time predictions over.
        model: Model name recorded on the result.
        experiment: Optional experiment name recorded on the result.
        repeats: Timed passes; the median per-sample latency is kept, so a single
            slow pass (GC, a cold cache) does not dominate.
        model_dir: Saved model directory whose on-disk size to record; ``None``
            leaves ``model_size_bytes`` unset.

    Returns:
        An :class:`EfficiencyRecord`. Latency is ``0.0`` for an empty batch.
    """
    batch = list(samples)
    latency_ms = _median_latency_ms(classifier, batch, repeats=repeats) if batch else 0.0
    return EfficiencyRecord(
        model=model,
        experiment=experiment,
        n_samples=len(batch),
        latency_ms=round(latency_ms, _LATENCY_DIGITS),
        model_size_bytes=None if model_dir is None else model_size_bytes(model_dir),
        n_params=count_parameters(getattr(classifier, "pipeline_", classifier)),
    )


def _median_latency_ms(
    classifier: DialectClassifier, samples: Sequence[Any], *, repeats: int
) -> float:
    """Median per-sample prediction latency in milliseconds over ``repeats`` passes."""
    per_sample: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        classifier.predict_samples(list(samples))
        elapsed = time.perf_counter() - start
        per_sample.append(elapsed / len(samples) * 1000.0)
    return float(np.median(per_sample))


def count_parameters(estimator: Any) -> int | None:
    """Best-effort parameter count of a fitted estimator or pipeline.

    Sums a linear model's coefficients (plus intercepts) or a torch module's
    tensor sizes. A pipeline is summed over its steps. Returns ``None`` when no
    step exposes a countable parameter set, rather than fabricating a number.

    Args:
        estimator: A fitted estimator, sklearn ``Pipeline``, or torch module.

    Returns:
        The parameter count, or ``None`` when it is not well defined.
    """
    steps = getattr(estimator, "steps", None)
    if steps is not None:
        counts = [count_parameters(step) for _, step in steps]
        present = [count for count in counts if count is not None]
        return sum(present) if present else None

    if hasattr(estimator, "coef_"):
        total = int(np.asarray(estimator.coef_).size)
        if hasattr(estimator, "intercept_"):
            total += int(np.asarray(estimator.intercept_).size)
        return total

    parameters = getattr(estimator, "parameters", None)
    if callable(parameters):
        try:
            return int(sum(int(tensor.numel()) for tensor in parameters()))
        except (TypeError, ValueError, AttributeError):
            return None
    return None


def model_size_bytes(model_dir: Path | str) -> int | None:
    """Total size in bytes of every file under a saved model directory.

    Args:
        model_dir: The directory a model was saved to.

    Returns:
        The summed file size, or ``None`` if the directory does not exist.
    """
    directory = Path(model_dir)
    if not directory.is_dir():
        return None
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def write_efficiency(records: Sequence[EfficiencyRecord], path: Path | str) -> None:
    """Write efficiency records as JSON (sorted keys), an excluded artifact.

    Sorted by ``(experiment, model)`` for a stable file order; the numbers inside
    remain machine dependent, so this file is never byte-compared.
    """
    ordered = sorted(records, key=lambda record: (record.experiment or "", record.model))
    write_sorted_json(Path(path), [record.model_dump(mode="json") for record in ordered])
