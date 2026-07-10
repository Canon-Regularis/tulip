"""Hierarchical family->dialect backoff classification.

A :class:`HierarchicalDialectClassifier` trains one
:class:`~tulip.pipeline.classifier.DialectClassifier` per label level and, at
prediction time, answers at the *finest* level it is confident about, backing
off to a coarser level otherwise. A confident sample gets a fine-grained
regional dialect; an ambiguous one is answered only at the family level rather
than guessed. The level at which each sample is answered therefore varies *per
sample* -- that variation is the whole point of the taxonomy note
(``taxonomy.py``) that "hierarchical classifiers can back off from fine-grained
to coarse-grained predictions".

Why this is *not* a ``DialectClassifier`` subclass (Liskov)
----------------------------------------------------------
``DialectClassifier.predict_batch`` carries a postcondition: every returned
:class:`~tulip.core.types.Prediction` has ``level == self.target`` -- one fixed
level for the whole batch. A backoff classifier deliberately breaks that: the
``level`` it returns differs from sample to sample. A subclass that weakened a
base-class postcondition would not be substitutable for its base, so this class
does **not** inherit from ``DialectClassifier``. It relates to it through
composition (it *owns* one per level) and satisfies the narrow
:class:`~tulip.pipeline.protocols.SamplePredictor` protocol instead, which fixes
only "one :class:`Prediction` per input :class:`Sample`, in order" -- a contract
a per-sample-varying level honours without strain.

Open/closed backoff policies
-----------------------------
Whether a prediction is "confident enough" to keep is decided by a
:class:`BackoffPolicy` -- a one-method protocol with small frozen-dataclass
implementations (:class:`ConfidenceThreshold`, :class:`MarginThreshold`,
:class:`NotAbstained`, the :class:`AlwaysAccept` null object, and the
:class:`AllOf`/:class:`AnyOf` combinators). Adding a new policy never edits this
module: policies are reconstructed from persisted artifacts through a
name-keyed factory (:func:`policy_from_spec`), not an ``if/elif`` chain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import ClassProbability, Prediction, TaskType
from tulip.labels.taxonomy import LabelLevel, family_for
from tulip.pipeline.classifier import DialectClassifier
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Self

    from tulip.core.types import Sample
    from tulip.pipeline.classifier import ComponentLike

_logger = get_logger(__name__)

#: Marker recorded in the save() sidecar so load() can reject foreign artifacts.
_ARTIFACT_KIND = "HierarchicalDialectClassifier"
#: File name of the hierarchical sidecar written alongside the per-level dirs.
_SIDECAR_NAME = "hierarchical.json"

__all__ = [
    "AllOf",
    "AlwaysAccept",
    "AnyOf",
    "BackoffPolicy",
    "ConfidenceThreshold",
    "HierarchicalConfig",
    "HierarchicalDialectClassifier",
    "MarginThreshold",
    "NotAbstained",
    "PolicySpec",
    "policy_from_spec",
]


# --------------------------------------------------------------- backoff policy


@runtime_checkable
class BackoffPolicy(Protocol):
    """Decides whether a prediction is trustworthy enough to keep.

    The single method keeps the protocol narrow (ISP): a policy is *only* a
    predicate over one prediction. Backoff — stepping to a coarser level when a
    policy rejects — is the classifier's job, not the policy's.
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
    behaves exactly like its finest-level classifier -- predictable and free of
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
    unless the policy also offers a ``to_spec`` -- reported here rather than
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


# --------------------------------------------------------------------- config


class HierarchicalConfig(BaseModel):
    """The persistable configuration of a hierarchical classifier.

    Module-owned rather than folded into
    :class:`~tulip.config.schemas.ExperimentConfig`, which is frozen and
    ``extra="forbid"`` -- backoff knobs cannot be bolted onto it without editing
    the frozen config contract (reported as friction, matching the precedent set
    by :class:`~tulip.pipeline.selftrain.SelfTrainConfig`).

    Attributes:
        levels: Label levels ordered coarse -> fine (e.g. family then dialect).
        policy: The backoff policy, stored as a reconstructable spec.
        mask_to_coarse: Whether fine predictions are constrained to the classes
            consistent with the coarse prediction before evaluation.
        seed: Seed applied before fitting every level's classifier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    levels: tuple[LabelLevel, ...]
    policy: PolicySpec
    mask_to_coarse: bool = True
    seed: int = 42


# ---------------------------------------------------------------- classifier


class HierarchicalDialectClassifier:
    """Predict at the finest confident level, backing off to coarser ones.

    Composes one :class:`~tulip.pipeline.classifier.DialectClassifier` per
    label level (never subclasses it -- see the module docstring for the Liskov
    argument) and satisfies :class:`~tulip.pipeline.protocols.SamplePredictor`.

    Args:
        levels: Label granularities ordered **coarse -> fine**, e.g.
            ``(LabelLevel.FAMILY, LabelLevel.DIALECT)``. At least two, all
            distinct.
        model: Model reference (registry name, mapping, or ComponentConfig),
            shared by every level's classifier.
        features: Feature component references; empty for raw-input models.
        task: Input modality, shared by every level.
        policy: Decides whether a fine prediction is kept or backed off;
            defaults to :class:`AlwaysAccept` (never back off).
        mask_to_coarse: When the coarser level is FAMILY and the finer is
            DIALECT, zero the dialect classes inconsistent with the coarse
            prediction and rescale the survivors to ``P(family) * P(dialect |
            family)`` before the policy evaluates the fine prediction. A
            projected distribution therefore sums to ``P(family)``, so a fine
            prediction can never claim more confidence than the coarse decision
            beneath it. If the predicted family has no dialects (``standard``),
            the dialect level cannot answer and the walk backs off. Silently
            inert for other level pairs.
        seed: Seed applied before fitting each level's classifier.

    Raises:
        ConfigurationError: if fewer than two levels are given, or they repeat.
    """

    def __init__(
        self,
        *,
        levels: Sequence[LabelLevel],
        model: ComponentLike,
        features: Sequence[ComponentLike] = (),
        task: TaskType = TaskType.TEXT,
        policy: BackoffPolicy | None = None,
        mask_to_coarse: bool = True,
        seed: int = 42,
    ) -> None:
        resolved = tuple(LabelLevel(level) for level in levels)
        if len(resolved) < 2:
            raise ConfigurationError(
                "levels must name at least two granularities ordered coarse to fine, "
                f"got {[level.value for level in resolved]}"
            )
        if len(set(resolved)) != len(resolved):
            raise ConfigurationError(
                f"levels must be unique, got {[level.value for level in resolved]}"
            )
        self.levels = resolved
        self.task = TaskType(task)
        self.policy: BackoffPolicy = policy if policy is not None else AlwaysAccept()
        self.mask_to_coarse = mask_to_coarse
        self.seed = seed

        self._model_ref = model
        self._feature_refs = tuple(features)
        #: Each level -> the level immediately coarser than it (absent for the
        #: coarsest), so masking can find its parent without index arithmetic.
        self._coarser_of = {resolved[i]: resolved[i - 1] for i in range(1, len(resolved))}
        self._fine_to_coarse = tuple(reversed(resolved))
        self._classifiers: dict[LabelLevel, DialectClassifier] = {}

    # ------------------------------------------------------------------ fit

    def fit(self, samples: Sequence[Sample]) -> Self:
        """Train one classifier per level on the same samples.

        Each level's classifier skips samples lacking that level's label (via
        :meth:`DialectClassifier.labelled_batch`), so partial labelling across
        the hierarchy is expected, not an error.

        Raises:
            DataError: if any level has nothing trainable (propagated from
                :meth:`DialectClassifier.fit`).
        """
        classifiers: dict[LabelLevel, DialectClassifier] = {}
        for level in self.levels:
            classifier = DialectClassifier(
                model=self._model_ref,
                features=self._feature_refs,
                task=self.task,
                target=level,
                seed=self.seed,
            )
            classifier.fit(samples)
            classifiers[level] = classifier
        self._classifiers = classifiers
        return self

    # -------------------------------------------------------------- predict

    def predict_samples(self, samples: Sequence[Sample]) -> list[Prediction]:
        """Classify each sample at the finest level its policy accepts.

        For every sample the walk starts at the finest level; if the policy
        accepts that (optionally coarse-masked) prediction it is returned,
        otherwise the walk steps one level coarser and retries. The coarsest
        level is always returned when nothing finer is accepted. Each returned
        :class:`Prediction` records the level it stopped at.

        Raises:
            ConfigurationError: if the classifier is not fitted.
            DataError: if any sample lacks this classifier's input modality
                (propagated from each level's classifier).
        """
        self._require_fitted()
        if not samples:
            return []
        by_level = {
            level: classifier.predict_samples(samples)
            for level, classifier in self._classifiers.items()
        }
        return [self._resolve(index, by_level) for index in range(len(samples))]

    def _resolve(self, index: int, by_level: dict[LabelLevel, list[Prediction]]) -> Prediction:
        """Walk fine -> coarse for one sample, returning the accepted level."""
        *backoff_levels, coarsest = self._fine_to_coarse
        for level in backoff_levels:
            candidate = self._candidate(level, index, by_level)
            # ``None`` means this level cannot express the coarser decision at all
            # (e.g. the family is ``standard``, which has no dialects). That is not
            # a low-confidence answer -- it is no answer -- so back off immediately.
            if candidate is not None and self.policy.accepts(candidate):
                return candidate
        coarsest_candidate = self._candidate(coarsest, index, by_level)
        assert coarsest_candidate is not None  # noqa: S101  # the coarsest level never projects
        return coarsest_candidate

    def _candidate(
        self, level: LabelLevel, index: int, by_level: dict[LabelLevel, list[Prediction]]
    ) -> Prediction | None:
        """Return the prediction at ``level``, projected onto its coarser neighbour.

        ``None`` signals that ``level`` cannot represent the coarser prediction.
        """
        prediction = by_level[level][index]
        coarser = self._coarser_of.get(level)
        if self.mask_to_coarse and level is LabelLevel.DIALECT and coarser is LabelLevel.FAMILY:
            return self._project_onto_family(prediction, by_level[coarser][index])
        return prediction

    def _project_onto_family(self, fine: Prediction, coarse: Prediction) -> Prediction | None:
        """Restrict the dialect distribution to the predicted family, by the chain rule.

        The dialect classes outside the coarse prediction's family drop to zero,
        and the survivors are rescaled to ``P(family) * P(dialect | family)``.

        Rescaling to the *coarse probability* rather than renormalising to 1.0 is
        the whole point. A family with exactly one dialect (Kashubian -> Kashubia)
        would otherwise renormalise to a certainty of 1.000 no matter how unsure
        the family classifier was, which silently defeats every confidence-based
        backoff policy. Under the chain rule that same case yields exactly
        ``P(family)``, so the fine prediction can never be more confident than the
        coarse decision it rests on. The returned distribution therefore sums to
        ``P(family)``, not to 1 -- it is a joint, not a conditional.

        Returns:
            The projected prediction, or ``None`` when the predicted family has no
            dialect children (``standard``), i.e. the finer level cannot answer.
        """
        if coarse.label is None:
            return fine  # the family classifier abstained; nothing to project onto
        consistent = {
            cp.label: cp.probability
            for cp in fine.probabilities
            if (family := family_for(cp.label)) is not None and family.value == coarse.label
        }
        total = sum(consistent.values())
        if total <= 0.0:
            _logger.debug(
                "family %r has no dialect children; the dialect level cannot answer",
                coarse.label,
            )
            return None
        family_probability = coarse.as_dict().get(coarse.label, 0.0)
        projected = tuple(
            ClassProbability(
                label=cp.label,
                probability=(
                    family_probability * consistent[cp.label] / total
                    if cp.label in consistent
                    else 0.0
                ),
            )
            for cp in fine.probabilities
        )
        top_label = max(consistent, key=lambda label: consistent[label])
        return Prediction(
            label=top_label,
            level=fine.level,
            probabilities=projected,
            abstained=fine.abstained,
        )

    # -------------------------------------------------------------- persist

    def save(self, path: Path | str) -> Path:
        """Persist every level's classifier plus the hierarchical config.

        Writes one subdirectory per level (``<path>/<level>``) via
        :meth:`DialectClassifier.save`, plus a ``hierarchical.json`` sidecar
        recording the level order, backoff policy, masking flag, and seed.

        Raises:
            ConfigurationError: if the classifier is unfitted, or its policy is
                not serialisable.
        """
        self._require_fitted()
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for level, classifier in self._classifiers.items():
            classifier.save(root / level.value)
        sidecar = {"kind": _ARTIFACT_KIND, "config": self._config().model_dump(mode="json")}
        payload = json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True)
        (root / _SIDECAR_NAME).write_text(payload + "\n", encoding="utf-8", newline="\n")
        _logger.info("saved hierarchical classifier (%d levels) to %s", len(self.levels), root)
        return root

    def _config(self) -> HierarchicalConfig:
        """Snapshot the persistable configuration of this classifier."""
        return HierarchicalConfig(
            levels=self.levels,
            policy=_spec_of(self.policy),
            mask_to_coarse=self.mask_to_coarse,
            seed=self.seed,
        )

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Restore a classifier saved by :meth:`save`, ready to predict.

        The per-level classifiers already persist their own model, features, and
        task, so those are read back from the loaded sub-classifiers rather than
        duplicated in the sidecar.

        Raises:
            DataError: if the artifact is missing, corrupt, or was not written
                by :meth:`save`.
        """
        root = Path(path)
        sidecar_path = root / _SIDECAR_NAME
        if not sidecar_path.is_file():
            raise DataError(f"no hierarchical classifier at {root}: missing {_SIDECAR_NAME}")
        try:
            raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataError(f"corrupt hierarchical sidecar at {sidecar_path}: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("kind") != _ARTIFACT_KIND:
            found = raw.get("kind") if isinstance(raw, dict) else None
            raise DataError(
                f"artifact at {root} was not saved by HierarchicalDialectClassifier.save() "
                f"(kind={found!r})"
            )
        try:
            config = HierarchicalConfig.model_validate(raw.get("config", {}))
        except ValidationError as exc:
            raise DataError(f"corrupt hierarchical config at {sidecar_path}: {exc}") from exc

        classifiers: dict[LabelLevel, DialectClassifier] = {}
        for level in config.levels:
            loaded = DialectClassifier.load(root / level.value)
            if loaded.target is not level:
                raise DataError(
                    f"sub-classifier under {level.value!r} was trained at "
                    f"{loaded.target.value!r}, not {level.value!r}"
                )
            classifiers[level] = loaded

        template = next(iter(classifiers.values()))
        instance = cls(
            levels=config.levels,
            model=template.model_config,
            features=template.feature_configs,
            task=template.task,
            policy=policy_from_spec(config.policy),
            mask_to_coarse=config.mask_to_coarse,
            seed=config.seed,
        )
        instance._classifiers = classifiers
        return instance

    def _require_fitted(self) -> None:
        if not self._classifiers:
            raise ConfigurationError(
                "this HierarchicalDialectClassifier is not fitted yet; call fit() first"
            )

    def __repr__(self) -> str:
        order = " -> ".join(level.value for level in self.levels)
        return (
            f"HierarchicalDialectClassifier(levels=[{order}], "
            f"policy={type(self.policy).__name__}, mask_to_coarse={self.mask_to_coarse})"
        )
