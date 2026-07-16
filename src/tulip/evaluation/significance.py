"""Statistical significance for the leaderboard: CIs and paired model tests.

A leaderboard that ranks 0.833 / 0.812 / 0.799 on one small split invites a false
reading: that the order is real. On n≈144 it may be noise. This module turns the
ranking into *claims a reader can trust*:

* **Bootstrap confidence intervals** per metric (seeded percentile bootstrap over
  a shared resample, so every model is judged on the same draws; common random
  numbers reduce comparison variance).
* **Exact McNemar tests** between every pair of models. Because the benchmark
  trains all competitors on the *identical* frozen split, their per-sample
  predictions are perfectly paired, which is exactly what McNemar needs; the
  exact binomial is computed with :func:`math.comb`, so no SciPy is required.
* **Holm-Bonferroni correction** across the pairwise tests, controlling the
  family-wise error rate that ad-hoc "compare everything" testing inflates.
* A **"tied with best"** grouping: models whose corrected difference from the top
  model is not significant, so a reader sees the true front-runner *set*.

Everything is deterministic: a fixed seed drives a fixed resample in fixed
order, so a committed significance report regenerates byte-for-byte.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import accuracy_score, f1_score

from tulip._serialize import round_floats
from tulip.core.exceptions import ConfigurationError
from tulip.evaluation._format import format_metric, markdown_table, write_sorted_json

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tulip.evaluation.predictions import SplitPredictions

__all__ = [
    "SIGNIFICANCE_FLOAT_DIGITS",
    "MetricCI",
    "ModelSignificance",
    "PairwiseTest",
    "SignificanceReport",
    "mcnemar_exact",
    "paired_significance",
]

#: Fixed rounding for persisted significance figures, so a committed report is
#: byte-identical across re-runs (mirrors ``PROVENANCE_FLOAT_DIGITS``).
SIGNIFICANCE_FLOAT_DIGITS = 6

#: Default number of bootstrap resamples for confidence intervals.
DEFAULT_RESAMPLES = 2000

#: Metrics carried with confidence intervals, computed exactly like
#: :func:`tulip.evaluation.metrics.compute_metrics` (macro/weighted over observed
#: classes, ``zero_division=0``) so the point estimates match the leaderboard.
_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
    "f1_macro": lambda y_true, y_pred: float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    ),
    "f1_weighted": lambda y_true, y_pred: float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ),
}


class MetricCI(BaseModel):
    """A metric's point estimate with a bootstrap confidence interval."""

    model_config = ConfigDict(frozen=True)

    metric: str
    point: float = Field(ge=0.0, le=1.0)
    low: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)


class ModelSignificance(BaseModel):
    """One model's confidence intervals plus whether it ties the best model."""

    model_config = ConfigDict(frozen=True)

    model: str
    metrics: tuple[MetricCI, ...]
    tied_with_best: bool


class PairwiseTest(BaseModel):
    """An exact McNemar comparison of two models on the identical paired split.

    Attributes:
        model_a / model_b: The two models, ``a`` being the better-ranked one.
        accuracy_delta: ``accuracy(a) - accuracy(b)`` on the full split.
        discordant_a: Samples ``a`` got right and ``b`` got wrong (McNemar ``b``).
        discordant_b: Samples ``b`` got right and ``a`` got wrong (McNemar ``c``).
        p_value: Two-sided exact McNemar p-value (unadjusted).
        p_value_holm: The p-value after Holm-Bonferroni correction over all pairs.
        significant: Whether ``p_value_holm`` is below the report's ``alpha``.
    """

    model_config = ConfigDict(frozen=True)

    model_a: str
    model_b: str
    accuracy_delta: float
    discordant_a: int = Field(ge=0)
    discordant_b: int = Field(ge=0)
    p_value: float = Field(ge=0.0, le=1.0)
    p_value_holm: float = Field(ge=0.0, le=1.0)
    significant: bool


class SignificanceReport(BaseModel):
    """Significance analysis for a set of models on one identical split."""

    model_config = ConfigDict(frozen=True)

    split: str
    n_samples: int = Field(ge=1)
    best_model: str
    ranking_metric: str
    alpha: float = Field(gt=0.0, lt=1.0)
    n_resamples: int = Field(ge=1)
    seed: int
    models: tuple[ModelSignificance, ...]
    pairwise: tuple[PairwiseTest, ...]

    def to_markdown(self) -> str:
        """Render the report as markdown: CI table, tie grouping, pairwise tests."""
        metric_names = [ci.metric for ci in self.models[0].metrics] if self.models else []
        headers = ("Model", *(_pretty(name) for name in metric_names), "Tied w/ best")
        ci_rows = [
            (
                model.model,
                *(_fmt_ci(ci) for ci in model.metrics),
                "yes" if model.tied_with_best else "no",
            )
            for model in self.models
        ]
        pair_rows = [
            (
                f"{test.model_a} vs {test.model_b}",
                format_metric(test.accuracy_delta),
                f"{test.discordant_a}/{test.discordant_b}",
                format_metric(test.p_value),
                format_metric(test.p_value_holm),
                "yes" if test.significant else "no",
            )
            for test in self.pairwise
        ]
        tied = [m.model for m in self.models if m.tied_with_best]
        return "\n\n".join(
            [
                f"# Significance: {self.split} (n={self.n_samples})",
                f"Best by {_pretty(self.ranking_metric)}: **{self.best_model}**. "
                f"Statistically tied with best (Holm-corrected McNemar, alpha={self.alpha}): "
                f"{', '.join(tied)}.",
                f"Confidence intervals are {int((1 - self.alpha) * 100)}% percentile "
                f"bootstrap ({self.n_resamples} resamples, seed {self.seed}).",
                markdown_table(headers, ci_rows),
                "## Pairwise McNemar tests (discordant a/b)",
                markdown_table(
                    ("Comparison", "Δ acc", "Discordant", "p", "p (Holm)", "sig."), pair_rows
                ),
            ]
        )

    def save(self, path: Path | str) -> None:
        """Write the report as deterministic JSON (sorted keys, rounded floats)."""
        write_sorted_json(
            Path(path), round_floats(self.model_dump(mode="json"), SIGNIFICANCE_FLOAT_DIGITS)
        )


def mcnemar_exact(
    correct_a: Sequence[bool] | np.ndarray, correct_b: Sequence[bool] | np.ndarray
) -> tuple[int, int, float]:
    """Two-sided exact McNemar test on two paired correctness vectors.

    Args:
        correct_a: Per-sample correctness of the first model.
        correct_b: Per-sample correctness of the second model, aligned with
            ``correct_a`` (same samples, same order).

    Returns:
        ``(discordant_a, discordant_b, p_value)`` where ``discordant_a`` is the
        count of samples ``a`` got right and ``b`` wrong (and vice versa), and
        ``p_value`` is the two-sided exact binomial McNemar p-value (``1.0`` when
        there are no discordant pairs).

    Raises:
        ConfigurationError: if the inputs differ in length or are empty.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ConfigurationError(
            f"correctness vectors must be the same length; got {a.shape} and {b.shape}"
        )
    if a.size == 0:
        raise ConfigurationError("cannot run McNemar on zero samples")
    discordant_a = int(np.sum(a & ~b))
    discordant_b = int(np.sum(~a & b))
    return discordant_a, discordant_b, _mcnemar_p(discordant_a, discordant_b)


def paired_significance(
    predictions: Sequence[SplitPredictions],
    *,
    ranking_metric: str = "f1_macro",
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> SignificanceReport:
    """Confidence intervals and paired McNemar tests over models on one split.

    Every :class:`SplitPredictions` must describe the *same* samples in the same
    order (the benchmark guarantees this by training all competitors on one
    frozen split). Models are ranked by ``ranking_metric``; the best is compared
    against every other with a Holm-corrected exact McNemar test, and the ones
    not significantly worse are flagged ``tied_with_best``.

    Args:
        predictions: One per model, all on the identical paired split.
        ranking_metric: Metric that decides the best model (a key of the
            reported metrics).
        alpha: Significance level for the tie grouping and CIs.
        n_resamples: Bootstrap resamples for the confidence intervals.
        seed: Seed for the (shared) bootstrap resample.

    Returns:
        A frozen :class:`SignificanceReport`.

    Raises:
        ConfigurationError: if fewer than two models are given, the samples are
            not aligned across models, or ``ranking_metric`` is unknown.
    """
    models = list(predictions)
    if len(models) < 2:
        raise ConfigurationError("significance analysis needs at least two models")
    if ranking_metric not in _METRICS:
        raise ConfigurationError(
            f"unknown ranking metric {ranking_metric!r}; expected one of {sorted(_METRICS)}"
        )
    ids = tuple(record.id for record in models[0].records)
    n = len(ids)
    if n == 0:
        raise ConfigurationError("cannot analyse significance on zero samples")
    for other in models[1:]:
        if tuple(record.id for record in other.records) != ids:
            raise ConfigurationError(
                f"model {other.model!r} is not aligned to {models[0].model!r}: significance "
                "testing requires identical samples in identical order across models"
            )

    y_true = np.asarray(models[0].true_labels(), dtype=object)
    preds = {model.model: np.asarray(model.pred_labels(), dtype=object) for model in models}
    correct = {model.model: model.correct() for model in models}

    # One shared resample matrix (common random numbers) drawn in fixed order.
    resample = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))

    scores = {name: _METRICS[ranking_metric](y_true, y_pred) for name, y_pred in preds.items()}
    ranked = sorted(scores, key=lambda name: (-scores[name], name))
    best = ranked[0]

    # All unordered pairs, better-ranked model first; Holm across their p-values.
    pairs = [(ranked[i], ranked[j]) for i in range(len(ranked)) for j in range(i + 1, len(ranked))]
    raw_p = [mcnemar_exact(correct[a], correct[b])[2] for a, b in pairs]
    holm_p = _holm(raw_p)

    pairwise = tuple(
        _pairwise_test(a, b, y_true, preds, correct, holm, alpha)
        for (a, b), holm in zip(pairs, holm_p, strict=True)
    )
    tied = _tied_with_best(best, pairwise, alpha)

    model_sig = tuple(
        ModelSignificance(
            model=name,
            metrics=_metric_cis(y_true, preds[name], resample, alpha),
            tied_with_best=name == best or name in tied,
        )
        for name in ranked
    )
    return SignificanceReport(
        split=models[0].split,
        n_samples=n,
        best_model=best,
        ranking_metric=ranking_metric,
        alpha=alpha,
        n_resamples=n_resamples,
        seed=seed,
        models=model_sig,
        pairwise=pairwise,
    )


def _metric_cis(
    y_true: np.ndarray, y_pred: np.ndarray, resample: np.ndarray, alpha: float
) -> tuple[MetricCI, ...]:
    """Percentile bootstrap CIs for every reported metric of one model."""
    lo_pct, hi_pct = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    cis = []
    for name, metric_fn in _METRICS.items():
        point = metric_fn(y_true, y_pred)
        draws = np.array([metric_fn(y_true[row], y_pred[row]) for row in resample])
        low, high = np.percentile(draws, [lo_pct, hi_pct])
        # A resample can never push a metric outside [point's] valid range, but
        # clamp against floating-point drift past [0, 1].
        cis.append(
            MetricCI(
                metric=name,
                point=_clip(point),
                low=_clip(min(float(low), point)),
                high=_clip(max(float(high), point)),
            )
        )
    return tuple(cis)


def _pairwise_test(
    model_a: str,
    model_b: str,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    correct: dict[str, np.ndarray],
    p_holm: float,
    alpha: float,
) -> PairwiseTest:
    """Assemble one :class:`PairwiseTest` from two models' correctness."""
    discordant_a, discordant_b, p_value = mcnemar_exact(correct[model_a], correct[model_b])
    delta = _METRICS["accuracy"](y_true, preds[model_a]) - _METRICS["accuracy"](
        y_true, preds[model_b]
    )
    return PairwiseTest(
        model_a=model_a,
        model_b=model_b,
        accuracy_delta=float(delta),
        discordant_a=discordant_a,
        discordant_b=discordant_b,
        p_value=p_value,
        p_value_holm=p_holm,
        significant=p_holm < alpha,
    )


def _tied_with_best(best: str, pairwise: Sequence[PairwiseTest], alpha: float) -> set[str]:
    """Models whose Holm-corrected difference from ``best`` is not significant."""
    tied: set[str] = set()
    for test in pairwise:
        if best not in (test.model_a, test.model_b):
            continue
        other = test.model_b if test.model_a == best else test.model_a
        if test.p_value_holm >= alpha:
            tied.add(other)
    return tied


def _mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    k = min(discordant_a, discordant_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def _holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment of ``p_values`` (order preserved)."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _clip(value: float) -> float:
    """Clamp a metric to [0, 1] against floating-point drift."""
    return min(1.0, max(0.0, float(value)))


def _pretty(metric: str) -> str:
    """Human-readable metric label for table headers."""
    return {"accuracy": "Accuracy", "f1_macro": "F1 (macro)", "f1_weighted": "F1 (weighted)"}.get(
        metric, metric
    )


def _fmt_ci(ci: MetricCI) -> str:
    """Render a metric CI as ``point [low, high]``."""
    return f"{format_metric(ci.point)} [{format_metric(ci.low)}, {format_metric(ci.high)}]"
