"""Hierarchical family->dialect backoff classification.

A :class:`HierarchicalDialectClassifier` trains one
:class:`~tulip.pipeline.classifier.DialectClassifier` per label level and, at
prediction time, answers at the *finest* level it is confident about, backing
off to a coarser level otherwise. A confident sample gets a fine-grained
regional dialect; an ambiguous one is answered only at the family level rather
than guessed. The level at which each sample is answered therefore varies *per
sample*; that variation is the whole point of the taxonomy note
(``taxonomy.py``) that "hierarchical classifiers can back off from fine-grained
to coarse-grained predictions".

Why this is *not* a ``DialectClassifier`` subclass
----------------------------------------------------------
``DialectClassifier.predict_batch`` carries a postcondition: every returned
:class:`~tulip.core.types.Prediction` has ``level == self.target``, one fixed
level for the whole batch. A backoff classifier deliberately breaks that: the
``level`` it returns differs from sample to sample. A subclass that weakened a
base-class postcondition would not be substitutable for its base, so this class
does **not** inherit from ``DialectClassifier``. It relates to it through
composition (it *owns* one per level) and satisfies the narrow
:class:`~tulip.pipeline.protocols.SamplePredictor` protocol instead, which fixes
only "one :class:`Prediction` per input :class:`Sample`, in order", a contract
a per-sample-varying level honours without strain.

Whether a prediction is "confident enough" to keep is decided by a
:class:`~tulip.pipeline.hierarchical.policies.BackoffPolicy`, a family of
predicates that lives in its own leaf module
(:mod:`tulip.pipeline.hierarchical.policies`) so that adding a policy never
edits this classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

from tulip._jsonio import read_json_object
from tulip._serialize import write_sorted_json
from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import Prediction, TaskType
from tulip.labels.taxonomy import LabelLevel
from tulip.pipeline.classifier import DialectClassifier
from tulip.pipeline.hierarchical.policies import (
    AlwaysAccept,
    PolicySpec,
    _spec_of,
    policy_from_spec,
)
from tulip.pipeline.hierarchical.projection import resolve_prediction
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
    ``extra="forbid"``: backoff knobs cannot be bolted onto it without editing
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
    label level (never subclasses it; see the module docstring for why) and
    satisfies :class:`~tulip.pipeline.protocols.SamplePredictor`.

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
        return [
            resolve_prediction(
                index,
                by_level,
                fine_to_coarse=self._fine_to_coarse,
                coarser_of=self._coarser_of,
                mask_to_coarse=self.mask_to_coarse,
                policy=self.policy,
            )
            for index in range(len(samples))
        ]

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
        write_sorted_json(root / _SIDECAR_NAME, sidecar)
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
        raw = read_json_object(sidecar_path, what="hierarchical sidecar")
        if raw.get("kind") != _ARTIFACT_KIND:
            raise DataError(
                f"artifact at {root} was not saved by HierarchicalDialectClassifier.save() "
                f"(kind={raw.get('kind')!r})"
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
