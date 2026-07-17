"""Probability-level fusion strategies for combining per-modality distributions.

A leaf module: it depends only on numpy and :mod:`tulip.core` (exceptions and
types), and must **not** import
:class:`~tulip.pipeline.classifier.DialectClassifier` or the composite
:class:`~tulip.pipeline.fusion.classifier.MultimodalClassifier`. Keeping the
strategies free of the classifier they serve is what lets them be unit-tested on
hand-built numpy stacks and reused by any future consumer.

Every :class:`FusionStrategy` maps a ``(n_modalities, n_samples, n_classes)``
stack of aligned probabilities, under a ``(n_modalities, n_samples)`` presence
mask, to one ``(n_samples, n_classes)`` distribution per sample. The identical
postcondition all three concrete strategies must honour (lone-modality
passthrough, per-row renormalisation, and rejecting a sample with no present
modality) is centralised in :class:`_FusionBase` so the strategies are provably
substitutable rather than each re-deriving it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np

from tulip.core.exceptions import ConfigurationError, DataError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

__all__ = [
    "ConfidenceWeightedFusion",
    "FusionStrategy",
    "LogarithmicPoolingFusion",
    "MaximumFusion",
    "WeightedAverageFusion",
    "build_strategy",
    "default_params",
]


@runtime_checkable
class FusionStrategy(Protocol):
    """Combines per-modality probabilities into one distribution per sample.

    ``stack`` has shape ``(n_modalities, n_samples, n_classes)``, every
    modality's probabilities already aligned to the same class columns, and
    ``mask`` has shape ``(n_modalities, n_samples)`` with ``True`` where that
    modality is present for that sample. The return has shape
    ``(n_samples, n_classes)`` with every row summing to 1.

    Postconditions every implementation must honour (verified by the
    parametrised contract test):

    * a sample with exactly one present modality passes that modality's
      distribution through unchanged;
    * every output row is renormalised to sum to 1 with no ``NaN``;
    * a sample with *no* present modality raises
      :class:`~tulip.core.exceptions.DataError`.
    """

    #: Stable identifier used to (de)serialise the strategy (see ``build_strategy``).
    kind: ClassVar[str]

    def fuse(self, stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fuse ``stack`` under ``mask`` into one distribution per sample."""


def _check_weights(weights: tuple[float, ...]) -> None:
    """Validate per-modality fusion weights (shared by the weighted strategies).

    Raises:
        ConfigurationError: if the weights are empty, non-finite, negative, or
            sum to zero (a strategy that ignores every modality is meaningless).
    """
    if not weights:
        raise ConfigurationError("fusion weights must be a non-empty sequence")
    array = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ConfigurationError(f"fusion weights must all be finite, got {weights!r}")
    if np.any(array < 0.0):
        raise ConfigurationError(f"fusion weights must be non-negative, got {weights!r}")
    if array.sum() <= 0.0:
        raise ConfigurationError(
            f"fusion weights must include at least one positive weight, got {weights!r}"
        )


class _FusionBase(abc.ABC):
    """Template enforcing the identical :class:`FusionStrategy` postcondition.

    Concrete strategies implement only :meth:`_pool`: how the *present*
    modalities combine into an unnormalised score. This template validates the
    shapes, rejects a sample with no present modality, forces a lone present
    modality to pass through unchanged, and renormalises every row. Centralising
    the contract here is exactly what makes the three strategies provably
    substitutable, rather than each re-deriving, and risking diverging from, the
    same postcondition.
    """

    #: Stable identifier used to (de)serialise the strategy; set by subclasses.
    kind: ClassVar[str]

    def config(self) -> dict[str, Any]:
        """Return JSON-serialisable parameters needed to rebuild this strategy."""
        return {}

    def fuse(self, stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """See :meth:`FusionStrategy.fuse`."""
        stack = np.asarray(stack, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if stack.ndim != 3:
            raise ConfigurationError(
                f"fusion stack must be 3-D (modalities, samples, classes), got shape {stack.shape}"
            )
        n_modalities, n_samples = stack.shape[0], stack.shape[1]
        if mask.shape != (n_modalities, n_samples):
            raise ConfigurationError(
                f"fusion mask shape {mask.shape} does not match the stack's "
                f"(modalities, samples) = {(n_modalities, n_samples)}"
            )
        present_counts = mask.sum(axis=0)
        absent = np.flatnonzero(present_counts == 0)
        if absent.size:
            raise DataError(
                f"{absent.size} sample(s) have no present modality to fuse "
                f"(first at column index {int(absent[0])})"
            )
        combined = self._pool(stack, mask)
        self._passthrough_singletons(stack, mask, combined, present_counts)
        return self._renormalise(combined)

    @staticmethod
    def _passthrough_singletons(
        stack: np.ndarray,
        mask: np.ndarray,
        combined: np.ndarray,
        present_counts: np.ndarray,
    ) -> None:
        """Overwrite single-modality rows with that modality's distribution, in place.

        Guarantees the "lone present modality passes through unchanged"
        postcondition for *every* strategy, including log pooling, whose
        single-expert pool ``p ** w`` would otherwise distort a weighted expert.
        """
        singletons = np.flatnonzero(present_counts == 1)
        if singletons.size:
            modality = np.argmax(mask[:, singletons], axis=0)
            combined[singletons] = stack[modality, singletons]

    @staticmethod
    def _renormalise(combined: np.ndarray) -> np.ndarray:
        """Scale every row to sum to 1, honouring the strategy's no-``NaN`` postcondition.

        In the shipping product each present modality's row already sums to 1, so
        row sums are positive. But a strategy is a public value object that a
        caller may hand a degenerate stack whose pooled row sums to zero (e.g. an
        all-zero probability column across every modality). Dividing that by its
        zero sum would yield ``NaN`` and break the documented postcondition, so a
        zero-sum row falls back to uniform, the same guard
        :func:`tulip.models.calibration` uses.
        """
        row_sums = combined.sum(axis=1, keepdims=True)
        degenerate = np.ravel(row_sums <= 0.0)
        if degenerate.any():
            combined = combined.copy()
            combined[degenerate] = 1.0
            row_sums = combined.sum(axis=1, keepdims=True)
        return combined / row_sums

    @abc.abstractmethod
    def _pool(self, stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Combine present modalities into an unnormalised ``(n_samples, n_classes)`` score.

        Implementations must exclude absent modalities (``mask[m, s]`` false)
        from sample ``s``. The template handles renormalisation and the
        lone-modality passthrough, so ``_pool`` need not.
        """


@dataclass(frozen=True)
class _WeightedFusion(_FusionBase):
    """Base for the weight-parametrised strategies (weighted average, log pool)."""

    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        _check_weights(self.weights)

    def config(self) -> dict[str, Any]:
        return {"weights": list(self.weights)}

    def _weight_array(self, n_modalities: int) -> np.ndarray:
        """Return the weights as an array, checked against the modality count."""
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.shape != (n_modalities,):
            raise ConfigurationError(
                f"{type(self).__name__} was given {weights.size} weight(s) but the "
                f"stack has {n_modalities} modality/ies"
            )
        return weights


@dataclass(frozen=True)
class WeightedAverageFusion(_WeightedFusion):
    """Fuse by the weighted arithmetic mean of the present modalities' probabilities.

    The linear opinion pool. With weights ``(1.0, 0.0)`` it reproduces the first
    modality exactly; a lone present modality passes through regardless of its
    weight.
    """

    kind: ClassVar[str] = "weighted_average"

    def _pool(self, stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
        weights = self._weight_array(stack.shape[0])
        effective = weights[:, None, None] * mask[:, :, None]
        return (effective * stack).sum(axis=0)


@dataclass(frozen=True)
class MaximumFusion(_FusionBase):
    """Fuse by the element-wise maximum across present modalities, then renormalise.

    A confident vote from either expert survives; the result is a proper
    distribution after renormalisation. Unweighted by construction.
    """

    kind: ClassVar[str] = "maximum"

    def _pool(self, stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.where(mask[:, :, None], stack, -np.inf)
        return masked.max(axis=0)


@dataclass(frozen=True)
class ConfidenceWeightedFusion(_FusionBase):
    """Fuse by a per-sample soft attention over modalities, weighted by confidence.

    Each present modality is weighted, for each sample independently, by its own
    top probability (its confidence on that sample), and the weights are
    normalised across modalities before the convex combination. So on a sample
    where the text expert is certain but the audio expert is unsure, text
    dominates the fused distribution, and vice versa. This is a lightweight,
    parameter-free attention over the two experts; it is not a learned multimodal
    transformer (which needs paired training data and a GPU), but it captures the
    same intuition that the more reliable modality should carry more weight where
    it is reliable. Unweighted by hyperparameter: the attention is data-driven.
    """

    kind: ClassVar[str] = "confidence"

    def _pool(self, stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
        confidence = stack.max(axis=2)  # (modalities, samples): each expert's top prob
        weights = confidence * mask  # absent modalities carry zero weight
        totals = weights.sum(axis=0, keepdims=True)
        # A zero total (both experts flat at zero confidence) is left to the base
        # renormalise/passthrough guards; avoid dividing by it here.
        safe_totals = np.where(totals > 0.0, totals, 1.0)
        normalised = weights / safe_totals
        return (normalised[:, :, None] * stack).sum(axis=0)


@dataclass(frozen=True)
class LogarithmicPoolingFusion(_WeightedFusion):
    """Fuse by the weighted geometric mean, the log-linear opinion pool.

    Computes ``prod_m p_m ** w_m`` over present modalities, renormalised.
    Probabilities are clipped to ``[EPS, 1]`` before the logarithm so a
    zero-probability class cannot produce ``-inf``; the pool is evaluated in log
    space with a per-row shift for numerical stability (equivalent up to the
    renormalisation that follows).
    """

    kind: ClassVar[str] = "logarithmic_pooling"

    #: Lower clip applied before the logarithm to avoid ``log(0)``.
    EPS: ClassVar[float] = 1e-12

    def _pool(self, stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
        weights = self._weight_array(stack.shape[0])
        logs = np.log(np.clip(stack, self.EPS, 1.0))
        effective = weights[:, None, None] * mask[:, :, None]
        weighted = (effective * logs).sum(axis=0)
        weighted -= weighted.max(axis=1, keepdims=True)  # stabilise exp; cancels on renorm
        return np.exp(weighted)


#: Strategy registry: a new named strategy registers here without editing any consumer.
_STRATEGY_REGISTRY: dict[str, Callable[[Mapping[str, Any]], FusionStrategy]] = {
    WeightedAverageFusion.kind: lambda params: WeightedAverageFusion(tuple(params["weights"])),
    MaximumFusion.kind: lambda _params: MaximumFusion(),
    ConfidenceWeightedFusion.kind: lambda _params: ConfidenceWeightedFusion(),
    LogarithmicPoolingFusion.kind: lambda params: LogarithmicPoolingFusion(
        tuple(params["weights"])
    ),
}

#: The strategy kinds that take a per-modality weight vector.
_WEIGHTED_KINDS: frozenset[str] = frozenset(
    {WeightedAverageFusion.kind, LogarithmicPoolingFusion.kind}
)

#: Equal split over two modalities, the default a weighted strategy takes when a
#: caller builds it without an explicit weight vector.
_DEFAULT_WEIGHTS: tuple[float, ...] = (0.5, 0.5)


def default_params(kind: str) -> dict[str, Any]:
    """Default build parameters for ``kind`` when a caller supplies none.

    A weighted strategy needs one weight per modality; absent an explicit choice it
    takes an equal split over two modalities. A parameter-free strategy needs
    nothing and returns an empty mapping. This keeps a caller from having to know
    which strategies are weighted.
    """
    return {"weights": list(_DEFAULT_WEIGHTS)} if kind in _WEIGHTED_KINDS else {}


def build_strategy(kind: str, params: Mapping[str, Any] | None = None) -> FusionStrategy:
    """Build a fusion strategy from a name and parameters (the deserialisation path).

    Args:
        kind: Registered strategy name (see :data:`_STRATEGY_REGISTRY`).
        params: Strategy parameters (e.g. ``{"weights": [0.5, 0.5]}``).

    Raises:
        ConfigurationError: if ``kind`` is unknown or its parameters are invalid.
    """
    try:
        factory = _STRATEGY_REGISTRY[kind]
    except KeyError:
        raise ConfigurationError(
            f"unknown fusion strategy {kind!r}; known: {sorted(_STRATEGY_REGISTRY)}"
        ) from None
    try:
        return factory(params or {})
    except KeyError as exc:
        raise ConfigurationError(
            f"fusion strategy {kind!r} is missing required parameter {exc}"
        ) from exc
