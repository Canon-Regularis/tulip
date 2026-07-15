"""Cross-corpus transfer evaluation: does the model learn dialect or channel?

A model trained and tested on one corpus can score well by keying on recording
channel, transcription style, or speaker set instead of dialect. The honest test
of generalisation is to train on some corpora and test on a different one. This
module runs two such protocols over the corpora already present in a sample set,
partitioned by :attr:`~tulip.core.types.Sample.source`.

* **Leave-one-corpus-out (LOCO).** For each corpus, train on every other corpus
  and test on the held-out one. This is the number that predicts deployment to a
  new region.
* **Transfer matrix.** Train on each corpus and test on each corpus, filling a
  full train-by-test grid. The off-diagonal cells show where a model transfers
  and where it does not; the diagonal is in-sample and optimistic.

Both reuse the training and evaluation machinery (``DialectClassifier``,
``evaluate_samples``, ``compute_metrics``); the pipeline imports are lazy, matching
:mod:`tulip.evaluation.leaderboard`, so there is no import cycle. Label-space
mismatch is handled by ``compute_metrics``: a dialect the training corpus never
saw degrades the score honestly instead of raising. Everything is seeded, so a
report is reproducible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import DataError
from tulip.evaluation._format import format_metric, markdown_table
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.config.schemas import ExperimentConfig
    from tulip.core.types import Sample

__all__ = [
    "CrossCorpusReport",
    "LocoResult",
    "TransferMatrix",
    "partition_by_source",
    "run_loco",
    "transfer_matrix",
]

_logger = get_logger(__name__)


class LocoResult(BaseModel):
    """One held-out corpus: train on the rest, test on it."""

    model_config = ConfigDict(frozen=True)

    held_out: str
    n_train: int = Field(ge=1)
    n_test: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)


class CrossCorpusReport(BaseModel):
    """Leave-one-corpus-out results across every corpus."""

    model_config = ConfigDict(frozen=True)

    model: str
    target: str
    results: tuple[LocoResult, ...]

    @property
    def macro_f1(self) -> float:
        """Mean held-out macro-F1 across corpora (0.0 when empty)."""
        if not self.results:
            return 0.0
        return sum(result.f1_macro for result in self.results) / len(self.results)

    def to_markdown(self) -> str:
        """Render the LOCO results as a markdown table."""
        rows = [
            (
                result.held_out,
                str(result.n_train),
                str(result.n_test),
                format_metric(result.accuracy),
                format_metric(result.f1_macro),
            )
            for result in self.results
        ]
        title = f"# Leave-one-corpus-out - {self.model} ({self.target})"
        summary = f"Mean held-out macro-F1: {format_metric(self.macro_f1)}"
        table = markdown_table(("Held-out corpus", "Train", "Test", "Accuracy", "F1 (macro)"), rows)
        return f"{title}\n\n{summary}\n\n{table}"


class TransferMatrix(BaseModel):
    """A train-corpus by test-corpus grid of macro-F1 scores."""

    model_config = ConfigDict(frozen=True)

    model: str
    target: str
    sources: tuple[str, ...]
    #: ``f1[train_source][test_source]`` macro-F1.
    f1: dict[str, dict[str, float]]

    def score(self, train_source: str, test_source: str) -> float:
        """Macro-F1 of training on ``train_source`` and testing on ``test_source``."""
        return self.f1[train_source][test_source]

    @property
    def mean_off_diagonal(self) -> float:
        """Mean transfer score across all train != test cells (0.0 if none)."""
        cells = [
            self.f1[train][test] for train in self.sources for test in self.sources if train != test
        ]
        return sum(cells) / len(cells) if cells else 0.0

    def to_markdown(self) -> str:
        """Render the matrix (rows = train corpus, columns = test corpus)."""
        headers = ("train \\ test", *self.sources)
        rows = [
            (train, *(format_metric(self.f1[train][test]) for test in self.sources))
            for train in self.sources
        ]
        title = f"# Cross-corpus transfer matrix - {self.model} ({self.target})"
        note = f"Mean off-diagonal (transfer) macro-F1: {format_metric(self.mean_off_diagonal)}"
        return f"{title}\n\n{note}\n\n{markdown_table(headers, rows)}"


def partition_by_source(samples: Sequence[Sample], target: str) -> dict[str, list[Sample]]:
    """Group samples that carry a ``target``-level label by their source corpus."""
    from tulip.labels.taxonomy import LabelLevel

    level = LabelLevel(target)
    by_source: dict[str, list[Sample]] = {}
    for sample in samples:
        if sample.labels.at_level(level) is not None:
            by_source.setdefault(sample.source, []).append(sample)
    return by_source


def run_loco(
    config: ExperimentConfig, *, samples: Sequence[Sample] | None = None
) -> CrossCorpusReport:
    """Run leave-one-corpus-out evaluation over the sources in the data.

    Args:
        config: The experiment declaration (data, features, model, target). Its
            ``split`` is ignored; the corpus partition defines the folds.
        samples: Pre-loaded samples to partition; loaded from ``config.data`` when
            omitted.

    Returns:
        A :class:`CrossCorpusReport`, one entry per held-out corpus.

    Raises:
        DataError: if fewer than two source corpora carry a target-level label.
    """
    from tulip.pipeline.experiment import build_classifier, evaluate_samples

    by_source = _require_multi_source(config, samples)
    sources = sorted(by_source)
    results: list[LocoResult] = []
    for held_out in sources:
        test = by_source[held_out]
        train = [sample for source in sources if source != held_out for sample in by_source[source]]
        classifier = build_classifier(config)
        classifier.fit(train)
        report = evaluate_samples(classifier, test, name=held_out)
        results.append(
            LocoResult(
                held_out=held_out,
                n_train=len(train),
                n_test=report.n_samples,
                accuracy=report.accuracy,
                f1_macro=report.f1_macro,
            )
        )
    _logger.info("LOCO %r: evaluated %d held-out corpora", config.model.name, len(results))
    return CrossCorpusReport(
        model=config.model.name, target=config.target.value, results=tuple(results)
    )


def transfer_matrix(
    config: ExperimentConfig, *, samples: Sequence[Sample] | None = None
) -> TransferMatrix:
    """Fill the full train-corpus by test-corpus macro-F1 matrix.

    Trains one classifier per source corpus and evaluates it against every source
    corpus, so each classifier is reused across its whole test row.

    Args:
        config: The experiment declaration.
        samples: Pre-loaded samples; loaded from ``config.data`` when omitted.

    Returns:
        A :class:`TransferMatrix`.

    Raises:
        DataError: if fewer than two source corpora carry a target-level label.
    """
    from tulip.pipeline.experiment import build_classifier, evaluate_samples

    by_source = _require_multi_source(config, samples)
    sources = sorted(by_source)
    grid: dict[str, dict[str, float]] = {}
    for train_source in sources:
        classifier = build_classifier(config)
        classifier.fit(by_source[train_source])
        grid[train_source] = {
            test_source: evaluate_samples(
                classifier, by_source[test_source], name=f"{train_source}->{test_source}"
            ).f1_macro
            for test_source in sources
        }
    _logger.info("transfer matrix %r: %dx%d", config.model.name, len(sources), len(sources))
    return TransferMatrix(
        model=config.model.name,
        target=config.target.value,
        sources=tuple(sources),
        f1=grid,
    )


def _require_multi_source(
    config: ExperimentConfig, samples: Sequence[Sample] | None
) -> dict[str, list[Sample]]:
    """Load and partition samples, requiring at least two source corpora."""
    if samples is None:
        from tulip.data.builder import DatasetBuilder

        samples = DatasetBuilder(config.data).load_samples()
    by_source = partition_by_source(samples, config.target.value)
    if len(by_source) < 2:
        raise DataError(
            f"cross-corpus evaluation needs >= 2 source corpora with a "
            f"{config.target.value!r} label; found {sorted(by_source) or 'none'}"
        )
    return by_source
