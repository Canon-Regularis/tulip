"""Backoff policies: the predicate family that decides "confident enough?".

Whether a fine-grained prediction is trustworthy enough to keep, rather than
back off to a coarser level, is a decision with its *own* reason to change:
new confidence heuristics get added, combined, and persisted independently of
how the classifier walks the taxonomy. That is why the policy family lives here,
apart from :mod:`tulip.pipeline.hierarchical.classifier`.

A policy is a one-method :class:`BackoffPolicy` predicate over a single
prediction: the small frozen-dataclass implementations
(:class:`ConfidenceThreshold`, :class:`MarginThreshold`, :class:`NotAbstained`,
the :class:`AlwaysAccept` null object, and the :class:`AllOf`/:class:`AnyOf`
combinators) never decide *when* to back off; that is the classifier's job.

Adding a new policy never edits the classifier: policies are reconstructed from
persisted artifacts through a name-keyed factory (:func:`policy_from_spec`), not
an ``if/elif`` chain.

A LEAF module: it depends only on pydantic and ``tulip.core`` and must never
import the classifier, so the dependency runs one way (classifier -> policies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from tulip.core.types import Prediction

__all__ = [
    "AllOf",
    "AlwaysAccept",
    "AnyOf",
    "BackoffPolicy",
    "ConfidenceThreshold",
    "MarginThreshold",
    "NotAbstained",
    "PolicySpec",
    "policy_from_spec",
]


# --------------------------------------------------------------- backoff policy


@runtime_checkable
class BackoffPolicy(Protocol):
    """Decides whether a prediction is trustworthy enough to keep.

    The single method keeps the protocol narrow: a policy is *only* a
    predicate over one prediction. Backoff, stepping to a coarser level when a
    policy rejects, is the classifier's job, not the policy's.
    """

    def accepts(self, prediction: Prediction) -> bool:
        """Return ``True`` to keep ``prediction``, ``False`` to back off."""
        ...


class PolicySpec(BaseModel):
    """Serialisable description of a :class:`BackoffPolicy`.

    A policy is a live object with behaviour; a spec is the flat, JSON-friendly
    record of *which* policy and with what parameters, so a fitted classifier
    round-trips through :meth:`HierarchicalDialectClassifier.save`. ``children``
    carries the sub-policies of the :class:`AllOf`/:class:`AnyOf` combinators.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    children: tuple[PolicySpec, ...] = ()


@dataclass(frozen=True)
class ConfidenceThreshold:
    """Accept only when the top class clears ``min_confidence``."""

    min_confidence: float

    def accepts(self, prediction: Prediction) -> bool:
        return prediction.confidence >= self.min_confidence

    def to_spec(self) -> PolicySpec:
        """Return this policy's serialisable spec (see :func:`policy_from_spec`)."""
        return PolicySpec(
            kind="confidence_threshold", params={"min_confidence": self.min_confidence}
        )


@dataclass(frozen=True)
class MarginThreshold:
    """Accept only when top1 - top2 probability clears ``min_margin``.

    A large margin means the top class stands clearly apart from its runner-up,
    a stricter signal than raw confidence when several classes are plausible.
    A single-class distribution is treated as an unbounded margin (accepted).
    """

    min_margin: float

    def accepts(self, prediction: Prediction) -> bool:
        probabilities = prediction.probabilities
        if len(probabilities) < 2:
            return True
        return (probabilities[0].probability - probabilities[1].probability) >= self.min_margin

    def to_spec(self) -> PolicySpec:
        """Return this policy's serialisable spec (see :func:`policy_from_spec`)."""
        return PolicySpec(kind="margin_threshold", params={"min_margin": self.min_margin})


@dataclass(frozen=True)
class NotAbstained:
    """Accept any prediction the underlying classifier did not abstain on."""

    def accepts(self, prediction: Prediction) -> bool:
        return not prediction.abstained

    def to_spec(self) -> PolicySpec:
        """Return this policy's serialisable spec (see :func:`policy_from_spec`)."""
        return PolicySpec(kind="not_abstained")


@dataclass(frozen=True)
class AlwaysAccept:
    """Null-object policy: keep every fine prediction, never back off.

    The default policy, so a hierarchical classifier with no policy configured
    behaves exactly like its finest-level classifier: predictable and free of
    scattered ``if policy is None`` checks in the backoff walk.
    """

    def accepts(self, prediction: Prediction) -> bool:
        return True

    def to_spec(self) -> PolicySpec:
        """Return this policy's serialisable spec (see :func:`policy_from_spec`)."""
        return PolicySpec(kind="always_accept")


@dataclass(frozen=True)
class AllOf:
    """Accept only when *every* sub-policy accepts (logical AND).

    An empty combinator accepts everything, mirroring ``all(())``.
    """

    policies: tuple[BackoffPolicy, ...] = ()

    def accepts(self, prediction: Prediction) -> bool:
        return all(policy.accepts(prediction) for policy in self.policies)

    def to_spec(self) -> PolicySpec:
        """Return this policy's serialisable spec (see :func:`policy_from_spec`)."""
        return PolicySpec(kind="all_of", children=tuple(_spec_of(p) for p in self.policies))


@dataclass(frozen=True)
class AnyOf:
    """Accept when *any* sub-policy accepts (logical OR).

    An empty combinator accepts nothing, mirroring ``any(())``.
    """

    policies: tuple[BackoffPolicy, ...] = ()

    def accepts(self, prediction: Prediction) -> bool:
        return any(policy.accepts(prediction) for policy in self.policies)

    def to_spec(self) -> PolicySpec:
        """Return this policy's serialisable spec (see :func:`policy_from_spec`)."""
        return PolicySpec(kind="any_of", children=tuple(_spec_of(p) for p in self.policies))


def _spec_of(policy: BackoffPolicy) -> PolicySpec:
    """Serialise ``policy``, or fail loudly if it is not persistable.

    Custom policies outside this module are welcome at prediction time (they
    satisfy :class:`BackoffPolicy`), but a classifier using one cannot be saved
    unless the policy also offers a ``to_spec``; reported here rather than
    writing a lossy artifact.
    """
    to_spec = getattr(policy, "to_spec", None)
    if callable(to_spec):
        result = to_spec()
        if isinstance(result, PolicySpec):
            return result
    raise ConfigurationError(
        f"backoff policy {type(policy).__name__!r} is not serialisable; it must implement "
        "to_spec() -> PolicySpec to be saved"
    )


def _build_confidence_threshold(spec: PolicySpec) -> BackoffPolicy:
    return ConfidenceThreshold(min_confidence=float(spec.params["min_confidence"]))


def _build_margin_threshold(spec: PolicySpec) -> BackoffPolicy:
    return MarginThreshold(min_margin=float(spec.params["min_margin"]))


def _build_not_abstained(spec: PolicySpec) -> BackoffPolicy:
    return NotAbstained()


def _build_always_accept(spec: PolicySpec) -> BackoffPolicy:
    return AlwaysAccept()


def _build_all_of(spec: PolicySpec) -> BackoffPolicy:
    return AllOf(policies=tuple(policy_from_spec(child) for child in spec.children))


def _build_any_of(spec: PolicySpec) -> BackoffPolicy:
    return AnyOf(policies=tuple(policy_from_spec(child) for child in spec.children))


#: Name-keyed factory for reconstructing policies. Adding a policy registers a
#: builder here; the classifier never grows an ``if/elif`` over policy kinds.
_POLICY_BUILDERS: dict[str, Callable[[PolicySpec], BackoffPolicy]] = {
    "confidence_threshold": _build_confidence_threshold,
    "margin_threshold": _build_margin_threshold,
    "not_abstained": _build_not_abstained,
    "always_accept": _build_always_accept,
    "all_of": _build_all_of,
    "any_of": _build_any_of,
}


def policy_from_spec(spec: PolicySpec) -> BackoffPolicy:
    """Rebuild a live :class:`BackoffPolicy` from its :class:`PolicySpec`.

    Raises:
        ConfigurationError: if ``spec.kind`` names no registered policy, or a
            required parameter is absent.
    """
    try:
        builder = _POLICY_BUILDERS[spec.kind]
    except KeyError:
        known = ", ".join(sorted(_POLICY_BUILDERS))
        raise ConfigurationError(
            f"unknown backoff policy kind {spec.kind!r}; known kinds: {known}"
        ) from None
    try:
        return builder(spec)
    except KeyError as exc:
        raise ConfigurationError(
            f"backoff policy {spec.kind!r} is missing required parameter {exc}"
        ) from exc
