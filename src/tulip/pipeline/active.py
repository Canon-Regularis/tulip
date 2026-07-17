"""Active learning: rank an unlabeled pool by how much labeling it would help.

Most Polish dialect corpora carry text or audio but no dialect labels; ``bigos``
is the motivating case. Given a classifier trained on a labeled seed set, this
module ranks the unlabeled samples an annotator should label first, so a fixed
annotation budget buys the most signal. It ranks only; turning a ranking into
labels is a human step, out of scope here.

An acquisition strategy scores each sample from the classifier's predicted
probabilities (higher score means more worth labeling). Three are classical
uncertainty measures: ``least_confidence``, ``margin``, and ``entropy``. The
fourth, ``intensity_gated``, is dialect-aware: it multiplies the uncertainty by
the text's dialect-intensity, so budget is not spent on standard Polish the model
merely happens to be unsure about.

Strategies are a registry, not a fixed enum, and each owns its own parameters.
Adding a strategy is a new class plus a decorator, with nothing central to edit,
and a strategy that needs tuning (``intensity_gated`` takes an intensity floor)
carries those knobs on its own constructor rather than on a shared config that
every strategy would have to grow. Ranking is a pure function of the fitted model
and the pool: identical inputs yield an identical ranking, ties broken by sample
id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip.core.registry import Registry
from tulip.pipeline._assembly import keep_with_raw

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tulip.core.types import Sample
    from tulip.pipeline.classifier import DialectClassifier

__all__ = [
    "STRATEGIES",
    "AcquisitionCandidate",
    "AcquisitionContext",
    "AcquisitionStrategy",
    "rank_for_labeling",
]

#: Scores and confidences are rounded to this many digits so a ranking report is
#: byte-stable when the content is.
ACQUISITION_FLOAT_DIGITS = 6

#: Clip floor for probabilities before taking a logarithm.
_LOG_FLOOR = 1e-12


@runtime_checkable
class AcquisitionStrategy(Protocol):
    """Scores a batch of predictions by how much labeling each would help."""

    name: str

    def score(self, context: AcquisitionContext) -> np.ndarray:
        """Return one non-negative score per sample; higher means label sooner."""


#: Canonical name -> acquisition strategy class. ``STRATEGIES.create(name,
#: **params)`` returns a ready strategy instance.
STRATEGIES: Registry[type[AcquisitionStrategy]] = Registry("acquisition strategy")


@dataclass(frozen=True)
class AcquisitionContext:
    """Everything a strategy scores over: the probabilities and the raw inputs.

    A frozen dataclass rather than a pydantic model because ``proba`` is a live
    numpy array with no meaningful validation schema.

    Attributes:
        proba: Probability matrix of shape ``(n_samples, n_classes)``, columns
            aligned with ``classes``.
        classes: The class labels, in column order.
        raws: The raw model inputs (texts for a text model), one per row of
            ``proba``. Uncertainty strategies ignore these; dialect-aware ones
            read them.
    """

    proba: np.ndarray
    classes: tuple[str, ...]
    raws: list[Any]


class AcquisitionCandidate(BaseModel):
    """One ranked unlabeled sample, with its score and the model's current guess."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    strategy: str
    score: float = Field(ge=0.0)
    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)


# ------------------------------------------------------------------ strategies


@STRATEGIES.register("least_confidence")
class LeastConfidence:
    """Score by ``1 - p(top class)``: the less sure the top call, the higher."""

    name = "least_confidence"

    def score(self, context: AcquisitionContext) -> np.ndarray:
        return 1.0 - context.proba.max(axis=1)


@STRATEGIES.register("margin")
class Margin:
    """Score by ``1 - (p1 - p2)``: a narrow top-two margin ranks higher."""

    name = "margin"

    def score(self, context: AcquisitionContext) -> np.ndarray:
        if context.proba.shape[1] < 2:
            return 1.0 - context.proba.max(axis=1)
        top_two = np.sort(context.proba, axis=1)[:, -2:]
        return 1.0 - (top_two[:, 1] - top_two[:, 0])


@STRATEGIES.register("entropy")
class Entropy:
    """Score by the Shannon entropy of the distribution, normalised to ``[0, 1]``."""

    name = "entropy"

    def score(self, context: AcquisitionContext) -> np.ndarray:
        return _normalised_entropy(context.proba)


@STRATEGIES.register("intensity_gated")
class IntensityGated:
    """Uncertainty gated by dialect intensity, so standard Polish is de-prioritised.

    The score is ``entropy * intensity`` where ``intensity`` is the text's
    overall dialectality in ``[0, 1)``. A sample must be both uncertain and
    genuinely dialectal to rank high, which keeps annotation budget off standard
    Polish the model happens to be unsure about. Text models only.

    Args:
        min_intensity: Intensity below this floor scores zero (a hard gate on top
            of the multiplicative one); ``0.0`` leaves the gate purely
            multiplicative.
        lexicon_path: Optional marker lexicon override for the intensity signal.
        rules_path: Optional isogloss rule override for the intensity signal.
    """

    name = "intensity_gated"

    def __init__(
        self,
        min_intensity: float = 0.0,
        lexicon_path: str | Path | None = None,
        rules_path: str | Path | None = None,
    ) -> None:
        self.min_intensity = min_intensity
        self.lexicon_path = lexicon_path
        self.rules_path = rules_path

    def score(self, context: AcquisitionContext) -> np.ndarray:
        from tulip.features.text.dialect_intensity import DialectIntensityExtractor

        texts = [_require_text(raw) for raw in context.raws]
        uncertainty = _normalised_entropy(context.proba)
        if not texts:
            return uncertainty
        extractor = DialectIntensityExtractor(self.lexicon_path, self.rules_path).fit(texts)
        intensity = extractor.transform(texts)[:, 0]  # the overall dialectality column
        gated = np.where(intensity >= self.min_intensity, intensity, 0.0)
        return uncertainty * gated


# ------------------------------------------------------------------ ranking


def rank_for_labeling(
    classifier: DialectClassifier,
    unlabeled: Sequence[Sample],
    *,
    strategy: str | AcquisitionStrategy = "entropy",
    budget: int | None = None,
) -> list[AcquisitionCandidate]:
    """Rank an unlabeled pool by acquisition score, most valuable first.

    Args:
        classifier: A fitted classifier; its task selects the modality read from
            each sample, and its ``predict_proba`` drives the scores.
        unlabeled: The candidate pool. A sample missing the classifier's input
            modality is skipped.
        strategy: A registered strategy name, or a strategy instance for one that
            needs tuning.
        budget: Keep only the top ``budget`` candidates; ``None`` keeps all.

    Returns:
        Candidates sorted by descending score, ties broken by sample id, capped
        at ``budget``.

    Raises:
        ConfigurationError: if ``budget`` is not positive, or a dialect-aware
            strategy is used on a non-text model.
        UnknownComponentError: if ``strategy`` names no registered strategy.
    """
    if budget is not None and budget < 1:
        from tulip.core.exceptions import ConfigurationError

        raise ConfigurationError(f"budget must be >= 1, got {budget}")

    scorer: AcquisitionStrategy = (
        STRATEGIES.create(strategy) if isinstance(strategy, str) else strategy
    )

    kept = keep_with_raw(unlabeled, classifier.task)
    if not kept:
        return []

    raws = [raw for _, raw in kept]
    proba = np.asarray(classifier.predict_proba(raws), dtype=np.float64)
    classes = tuple(str(label) for label in classifier.classes_)
    scores = np.asarray(scorer.score(AcquisitionContext(proba, classes, raws)), dtype=np.float64)

    candidates = [
        AcquisitionCandidate(
            sample_id=sample.id,
            strategy=scorer.name,
            score=round(float(max(score, 0.0)), ACQUISITION_FLOAT_DIGITS),
            predicted_label=classes[int(np.argmax(row))],
            confidence=round(float(row[int(np.argmax(row))]), ACQUISITION_FLOAT_DIGITS),
        )
        for (sample, _), score, row in zip(kept, scores, proba, strict=True)
    ]
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.sample_id))
    if budget is not None:
        candidates = candidates[:budget]
    return candidates


# ------------------------------------------------------------------ helpers


def _normalised_entropy(proba: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy normalised to ``[0, 1]`` by ``log(n_classes)``."""
    n_classes = proba.shape[1]
    if n_classes < 2:
        return np.zeros(proba.shape[0], dtype=np.float64)
    clipped = np.clip(proba, _LOG_FLOOR, 1.0)
    entropy = -(proba * np.log(clipped)).sum(axis=1)
    return entropy / np.log(n_classes)


def _require_text(raw: Any) -> str:
    """Coerce a raw input to text, rejecting a non-text (e.g. audio) model."""
    if not isinstance(raw, str):
        from tulip.core.exceptions import ConfigurationError

        raise ConfigurationError(
            "the intensity_gated strategy needs text inputs; this classifier's task is not text"
        )
    return raw
