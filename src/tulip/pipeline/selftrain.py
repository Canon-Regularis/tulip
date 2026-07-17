"""Semi-supervised self-training (pseudo-labeling) for unlabeled corpora.

Some corpora carry substantial audio/text volume but no dialect labels.
``bigos`` (catalog ``label_levels=()``) is the motivating case. Self-training
turns that volume into training signal: a classifier trained on the labeled
seed set labels the unlabeled pool, its most confident guesses become
*pseudo-labeled* samples, and a fresh classifier is refit on the union. The
loop repeats until nothing confident is left to add or the iteration budget is
spent.

The knobs live in a module-owned :class:`SelfTrainConfig` rather than on
:class:`~tulip.config.schemas.ExperimentConfig`: that schema is frozen and
``extra="forbid"``, so self-training parameters cannot be bolted onto it
without editing the frozen config contract (reported as friction). Everything
here is fully seeded and deterministic: identical inputs and seed yield
identical pseudo counts and an identical final classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import DataError
from tulip.core.types import DialectLabels, Prediction, Sample, TaskType
from tulip.labels.taxonomy import LabelLevel
from tulip.pipeline._assembly import keep_with_raw
from tulip.pipeline.classifier import ComponentLike, DialectClassifier
from tulip.utils.logging import get_logger
from tulip.utils.seed import set_global_seed

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = get_logger(__name__)


class SelfTrainConfig(BaseModel):
    """Parameters governing one self-training run.

    This is a standalone, module-owned schema, *not* an extension of
    :class:`~tulip.config.schemas.ExperimentConfig`, which is frozen and
    forbids extra fields.

    Attributes:
        confidence_threshold: Minimum top-class probability for a prediction to
            be trusted as a pseudo-label. Higher values keep fewer, cleaner
            labels; the default is deliberately conservative.
        max_iterations: Upper bound on self-training rounds. The loop also stops
            early once a round adds nothing new (convergence).
        max_pseudo_per_iter: Cap on pseudo-labels admitted per round; the most
            confident survive. ``None`` admits every prediction above the
            threshold.
        target: Label granularity to train and pseudo-label at.
        task: Input modality; selects the sample field (``text`` vs
            ``audio_path``) used as the raw model input. Kept off the design
            sketch but required so audio-bearing unlabeled corpora (e.g.
            ``bigos``) are reachable, not just text.
        seed: Seed applied before every fit, making the whole run reproducible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    max_iterations: int = Field(default=3, ge=1)
    max_pseudo_per_iter: int | None = Field(default=None, ge=1)
    target: LabelLevel = LabelLevel.DIALECT
    task: TaskType = TaskType.TEXT
    seed: int = 42


@dataclass(frozen=True)
class SelfTrainResult:
    """The outcome of a self-training run.

    A frozen dataclass rather than a pydantic model: the final ``classifier``
    is a live, fitted object that has no meaningful validation schema, and
    embedding it in a pydantic model would force arbitrary-type escape hatches
    for no benefit.

    Attributes:
        iterations: Number of rounds that actually added pseudo-labels (rounds
            that added nothing trigger early stopping and are not counted).
        n_pseudo_per_iteration: New pseudo-labels admitted in each counted
            round, in order. Every entry is positive; the running total is
            non-decreasing and its sum equals ``len(pseudo_samples)``.
        classifier: The final classifier, fitted on the labeled seed set plus
            all accumulated pseudo-labels. Ready to :meth:`predict`.
        pseudo_samples: Every pseudo-labeled :class:`Sample` created, each a
            fresh copy carrying ``metadata["pseudo_labeled"] = True``.
    """

    iterations: int
    n_pseudo_per_iteration: tuple[int, ...]
    classifier: DialectClassifier
    pseudo_samples: tuple[Sample, ...]


def _fit_fresh(
    *,
    model: ComponentLike,
    features: Sequence[ComponentLike],
    config: SelfTrainConfig,
    samples: Sequence[Sample],
) -> DialectClassifier:
    """Build and fit a brand-new classifier (never an incremental refit)."""
    classifier = DialectClassifier(
        model=model,
        features=features,
        task=config.task,
        target=config.target,
        seed=config.seed,
    )
    return classifier.fit(samples)


def _make_pseudo_sample(sample: Sample, prediction: Prediction, *, target: LabelLevel) -> Sample:
    """Return a fresh pseudo-labeled copy of ``sample`` (the original is untouched).

    The copy carries a single label at ``target`` (family is auto-derived by
    :class:`DialectLabels` when the target is ``dialect``) and records its
    provenance in metadata so pseudo-labels remain auditable and separable
    from gold ones.
    """
    labels = DialectLabels(**{target.value: prediction.label})
    metadata = {
        **sample.metadata,
        "pseudo_labeled": True,
        "confidence": prediction.confidence,
    }
    return sample.model_copy(update={"labels": labels, "metadata": metadata})


def self_train(
    *,
    labeled: Sequence[Sample],
    unlabeled: Sequence[Sample],
    model: ComponentLike,
    features: Sequence[ComponentLike] = (),
    config: SelfTrainConfig | None = None,
) -> SelfTrainResult:
    """Grow a classifier from a labeled seed set using confident pseudo-labels.

    A fresh classifier is fit on ``labeled``; it then labels the unlabeled pool,
    the predictions that clear ``confidence_threshold`` become pseudo-labeled
    samples (capped and ranked deterministically by descending confidence, ties
    broken by sample id), and a *fresh* classifier is refit on the labeled set
    plus every pseudo-label accumulated so far. The loop repeats up to
    ``max_iterations`` and stops early the moment a round admits nothing new.

    Args:
        labeled: The gold-labeled seed corpus; must yield at least one sample
            trainable at ``config.target`` for the configured modality.
        unlabeled: Candidate samples without usable labels (e.g. ``bigos``).
            Samples lacking the modality's raw input are ignored.
        model: Model reference (registry name, mapping, or ComponentConfig).
        features: Feature component references; empty for raw-input models.
        config: Self-training parameters; defaults to :class:`SelfTrainConfig`.

    Returns:
        A :class:`SelfTrainResult` with per-iteration pseudo counts, the created
        pseudo-samples, and the final fitted classifier.

    Raises:
        DataError: if ``labeled`` has nothing trainable at ``config.target``.
    """
    config = config or SelfTrainConfig()
    set_global_seed(config.seed)

    if not labeled:
        raise DataError("self_train requires a non-empty labeled seed set")

    # Precompute each unlabeled sample's raw input once; drop those without the
    # modality (they can never be predicted or pseudo-labeled).
    candidates = keep_with_raw(unlabeled, config.task)
    if len(candidates) < len(unlabeled):
        _logger.info(
            "self_train: %d/%d unlabeled samples lack %s input and are skipped",
            len(unlabeled) - len(candidates),
            len(unlabeled),
            config.task.value,
        )

    classifier = _fit_fresh(model=model, features=features, config=config, samples=labeled)

    pseudo_samples: list[Sample] = []
    pseudo_ids: set[str] = set()
    counts: list[int] = []

    for iteration in range(1, config.max_iterations + 1):
        remaining = [(s, raw) for s, raw in candidates if s.id not in pseudo_ids]
        if not remaining:
            break

        predictions = classifier.predict_batch([raw for _, raw in remaining])
        accepted = [
            (sample, prediction)
            for (sample, _), prediction in zip(remaining, predictions, strict=True)
            if not prediction.abstained
            and prediction.label is not None
            and prediction.confidence >= config.confidence_threshold
        ]
        # Deterministic order: most confident first, sample id breaks ties.
        accepted.sort(key=lambda item: (-item[1].confidence, item[0].id))
        if config.max_pseudo_per_iter is not None:
            accepted = accepted[: config.max_pseudo_per_iter]

        if not accepted:
            _logger.info("self_train: iteration %d added no pseudo-labels; converged", iteration)
            break

        for sample, prediction in accepted:
            pseudo_samples.append(_make_pseudo_sample(sample, prediction, target=config.target))
            pseudo_ids.add(sample.id)
        counts.append(len(accepted))
        _logger.info(
            "self_train: iteration %d added %d pseudo-labels (%d total)",
            iteration,
            len(accepted),
            len(pseudo_samples),
        )

        classifier = _fit_fresh(
            model=model,
            features=features,
            config=config,
            samples=[*labeled, *pseudo_samples],
        )

    return SelfTrainResult(
        iterations=len(counts),
        n_pseudo_per_iteration=tuple(counts),
        classifier=classifier,
        pseudo_samples=tuple(pseudo_samples),
    )


__all__ = ["SelfTrainConfig", "SelfTrainResult", "self_train"]
