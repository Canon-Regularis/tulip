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

Whether a prediction is "confident enough" to keep is decided by a
:class:`~tulip.pipeline.hierarchical.policies.BackoffPolicy`, an open/closed
family of predicates that lives in its own leaf module
(:mod:`tulip.pipeline.hierarchical.policies`) so that adding a policy never
edits this classifier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import ClassProbability, Prediction, TaskType
from tulip.labels.taxonomy import LabelLevel, family_for
from tulip.pipeline.classifier import DialectClassifier
from tulip.pipeline.hierarchical.policies import (
    AlwaysAccept,
    PolicySpec,
    _spec_of,
    policy_from_spec,
)
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from tulip.core.types import Sample
    from tulip.pipeline.classifier import ComponentLike
    from tulip.pipeline.hierarchical.policies import BackoffPolicy

_logger = get_logger(__name__)

#: Marker recorded in the save() sidecar so load() can reject foreign artifacts.
_ARTIFACT_KIND = "HierarchicalDialectClassifier"
#: File name of the hierarchical sidecar written alongside the per-level dirs.
_SIDECAR_NAME = "hierarchical.json"

__all__ = [
    "HierarchicalConfig",
    "HierarchicalDialectClassifier",
]


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
