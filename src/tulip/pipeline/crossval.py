"""Grouped, stratified K-fold cross-validation with multi-seed aggregation.

A single train/test split gives one number with no error bar. On the small,
single-locality corpora tulip targets, that number swings with the split.
Cross-validation replaces it with a mean and a confidence interval.

The folds are speaker-disjoint and label-stratified, the same discipline as the
main split (:mod:`tulip.data.splitting`): a speaker never appears in both the
training and test side of a fold, so a fold cannot reward speaker
re-identification. The folding itself reuses scikit-learn's
:class:`~sklearn.model_selection.StratifiedGroupKFold` rather than
reimplementing it.

Run it over several seeds to average out the fold assignment. Every metric is
summarised by its mean, standard deviation, and a percentile confidence interval
across all ``k * len(seeds)`` fold runs. Everything is seeded, so a report is
reproducible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import DataError
from tulip.evaluation._format import format_metric, markdown_table
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tulip.config.schemas import ExperimentConfig
    from tulip.core.types import Sample
    from tulip.labels.taxonomy import LabelLevel

__all__ = [
    "CVConfig",
    "CVFoldResult",
    "CVReport",
    "MetricSummary",
    "grouped_stratified_kfold",
    "run_cross_validation",
]

_logger = get_logger(__name__)

#: Metrics summarised in a CV report, read from each fold's EvaluationReport.
_METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted")


class CVConfig(BaseModel):
    """How to cross-validate: fold count, seeds, and the grouping key.

    Owned by this module, not layered onto the frozen ``ExperimentConfig``: it is
    an evaluation protocol, not part of a single experiment declaration.
    """

    model_config = ConfigDict(frozen=True)

    k: int = Field(default=5, ge=2)
    seeds: tuple[int, ...] = Field(default=(0,), min_length=1)
    group_by: str = "speaker_id"

    @property
    def n_runs(self) -> int:
        """Total fold runs: ``k`` folds for each seed."""
        return self.k * len(self.seeds)


class CVFoldResult(BaseModel):
    """One fold's metrics for one seed."""

    model_config = ConfigDict(frozen=True)

    seed: int
    fold: int
    n_train: int = Field(ge=0)
    n_test: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)
    f1_weighted: float = Field(ge=0.0, le=1.0)


class MetricSummary(BaseModel):
    """A metric aggregated across all fold runs."""

    model_config = ConfigDict(frozen=True)

    metric: str
    mean: float = Field(ge=0.0, le=1.0)
    std: float = Field(ge=0.0)
    low: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)


class CVReport(BaseModel):
    """Cross-validation summary for one model on one dataset."""

    model_config = ConfigDict(frozen=True)

    model: str
    target: str
    k: int = Field(ge=2)
    seeds: tuple[int, ...]
    metrics: tuple[MetricSummary, ...]
    folds: tuple[CVFoldResult, ...]

    def summary(self, metric: str = "f1_macro") -> MetricSummary:
        """Return the aggregate for one metric.

        Raises:
            KeyError: if ``metric`` was not summarised.
        """
        for entry in self.metrics:
            if entry.metric == metric:
                return entry
        raise KeyError(metric)

    def to_markdown(self) -> str:
        """Render the report as a markdown metrics table (mean and 95% CI)."""
        rows = [
            (
                entry.metric,
                format_metric(entry.mean),
                format_metric(entry.std),
                f"[{format_metric(entry.low)}, {format_metric(entry.high)}]",
            )
            for entry in self.metrics
        ]
        title = (
            f"# Cross-validation - {self.model} ({self.target})\n\n"
            f"{self.k}-fold, seeds {list(self.seeds)}, {len(self.folds)} runs"
        )
        return f"{title}\n\n{markdown_table(('Metric', 'Mean', 'Std', '95% CI'), rows)}"


def grouped_stratified_kfold(
    samples: Sequence[Sample],
    *,
    k: int,
    seed: int,
    target: LabelLevel,
    group_by: str = "speaker_id",
) -> Iterator[tuple[list[Sample], list[Sample]]]:
    """Yield ``k`` speaker-disjoint, label-stratified ``(train, test)`` folds.

    Samples without a label at ``target`` are dropped. Grouping is by
    ``group_by`` (default ``speaker_id``); a sample missing that key is its own
    group, so it can never leak across a fold boundary.

    Args:
        samples: The samples to fold.
        k: Number of folds.
        seed: Seed for the fold assignment.
        target: Label level to stratify on.
        group_by: Sample attribute that groups rows (kept together in a fold).

    Yields:
        ``(train_samples, test_samples)`` for each fold, in fold order.

    Raises:
        DataError: if fewer than ``k`` labelled samples or distinct groups exist.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    labelled = [s for s in samples if s.labels.at_level(target) is not None]
    if len(labelled) < k:
        raise DataError(f"need at least k={k} labelled samples for CV, got {len(labelled)}")
    labels = [str(s.labels.at_level(target)) for s in labelled]
    groups = [str(getattr(s, group_by, None) or s.id) for s in labelled]
    n_groups = len(set(groups))
    if n_groups < k:
        raise DataError(
            f"only {n_groups} distinct {group_by!r} group(s); a {k}-fold "
            "speaker-disjoint CV needs at least that many"
        )
    placeholder = np.zeros((len(labelled), 1))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(placeholder, labels, groups):
        yield [labelled[i] for i in train_index], [labelled[i] for i in test_index]


def run_cross_validation(config: ExperimentConfig, cv: CVConfig, *, n_jobs: int = 1) -> CVReport:
    """Cross-validate the experiment's model and aggregate the fold metrics.

    Loads and prepares the corpus once (load, clean, deduplicate), then for every
    seed folds it, trains a fresh classifier per fold, and evaluates on the held
    out fold. Deduplication runs before folding, so near-duplicates cannot
    straddle a fold.

    Args:
        config: The experiment declaration (data, features, model, target).
        cv: The cross-validation protocol.
        n_jobs: Fold runs to execute in parallel. ``1`` runs in-process and
            sequentially (the default). Above 1 (or ``-1`` for all cores) runs
            folds in separate processes via joblib; because each fold's fit
            re-seeds its own process, the aggregated report is identical to the
            sequential run.

    Returns:
        A :class:`CVReport` with per-metric aggregates and per-fold results.

    Raises:
        DataError: if the corpus cannot be folded (see
            :func:`grouped_stratified_kfold`).
    """
    from joblib import Parallel, delayed

    from tulip.data.builder import DatasetBuilder

    samples = DatasetBuilder(config.data).load_samples()
    # Materialise every (seed, fold) unit up front so they can be dispatched
    # together; joblib preserves this order, so the aggregate is deterministic.
    tasks = [
        (seed, fold, train, test)
        for seed in cv.seeds
        for fold, (train, test) in enumerate(
            grouped_stratified_kfold(
                samples, k=cv.k, seed=seed, target=config.target, group_by=cv.group_by
            )
        )
    ]
    fold_results: list[CVFoldResult] = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_fold)(config, seed, fold, train, test) for seed, fold, train, test in tasks
    )
    _logger.info(
        "cross-validation %r: %d runs (%d-fold x %d seeds)",
        config.model.name,
        len(fold_results),
        cv.k,
        len(cv.seeds),
    )
    return CVReport(
        model=config.model.name,
        target=config.target.value,
        k=cv.k,
        seeds=cv.seeds,
        metrics=tuple(_summarise(metric, fold_results) for metric in _METRICS),
        folds=tuple(fold_results),
    )


def _run_one_fold(
    config: ExperimentConfig,
    seed: int,
    fold: int,
    train: Sequence[Sample],
    test: Sequence[Sample],
) -> CVFoldResult:
    """Train and evaluate one fold (a parallel unit).

    A module-level function so joblib's process backend can pickle it. The model
    is seeded from the fold's ``seed`` so a fold's result is independent of which
    process runs it.
    """
    from tulip.pipeline.experiment import build_classifier, evaluate_samples

    candidate = config.model_copy(update={"seed": seed})
    classifier = build_classifier(candidate)
    classifier.fit(train)
    report = evaluate_samples(classifier, test, name=f"seed{seed}-fold{fold}")
    return CVFoldResult(
        seed=seed,
        fold=fold,
        n_train=len(train),
        n_test=report.n_samples,
        accuracy=report.accuracy,
        balanced_accuracy=report.balanced_accuracy,
        f1_macro=report.f1_macro,
        f1_weighted=report.f1_weighted,
    )


def _summarise(metric: str, folds: Sequence[CVFoldResult]) -> MetricSummary:
    """Aggregate one metric across fold runs into mean, std, and a 95% CI."""
    values = np.array([getattr(fold, metric) for fold in folds], dtype=float)
    low, high = np.percentile(values, [2.5, 97.5]) if values.size > 1 else (values[0], values[0])
    return MetricSummary(
        metric=metric,
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        low=float(low),
        high=float(high),
    )
