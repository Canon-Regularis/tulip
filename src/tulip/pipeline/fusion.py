"""Late-fusion of a text and an audio classifier over one :class:`Sample` stream.

A :class:`~tulip.pipeline.classifier.DialectClassifier` fixes ``task`` to
TEXT *xor* AUDIO, so the two modalities that a :class:`~tulip.core.types.Sample`
carries (``text`` and ``audio_path``) never combine, even though many corpora
provide both. This module fuses them at the probability level.

**Why not extend the task enum.** ``TaskType`` (in the frozen ``core/types``)
has only ``TEXT`` and ``AUDIO``; adding a ``MULTIMODAL`` member would be the
wrong design even if the enum were editable. A multimodal model is not a third
*modality* -- it is a *composition* of two single-modality models, each an
expert on its own input. Modelling it as two :class:`DialectClassifier`s joined
by a fusion strategy keeps each base independently trainable, swappable, and
persistable, and lets a caller weight or replace either side without touching
the other. The frozen enum is therefore reported as friction that this design
routes around rather than fights (see the task notes / ``docs/architecture.md``).

**Why this is not a DialectClassifier subclass (LSP).**
``DialectClassifier.predict_batch`` guarantees two postconditions:
every returned :class:`~tulip.core.types.Prediction` has ``level == self.target``
*and* its inputs are raw values of one modality. A multimodal classifier reads
*both* modalities from a whole :class:`Sample`, so it cannot honour the second
postcondition and must not be substitutable for a ``DialectClassifier``.
:class:`MultimodalClassifier` therefore relates to its bases by *composition*
and exposes only the narrow
:class:`~tulip.pipeline.protocols.SamplePredictor` contract, never inheritance.

**Why the dependency is a structural type (DIP).** The classifier depends on
:class:`ProbabilisticClassifier` -- a narrow structural protocol requiring just
``classes_``, ``target``, ``task``, and ``predict_proba`` -- not on the concrete
``DialectClassifier``. Any object exposing those satisfies it, which is what
lets tests inject a cheap deterministic audio stub instead of training a real
speech model, keeping the suite hermetic and fast.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import ClassProbability, Prediction, TaskType
from tulip.pipeline.classifier import DialectClassifier
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any, Self

    from tulip.core.types import Sample
    from tulip.labels.taxonomy import LabelLevel

_logger = get_logger(__name__)


@runtime_checkable
class ProbabilisticClassifier(Protocol):
    """The slice of a classifier that late fusion needs: aligned probabilities.

    A deliberately narrow structural type (DIP/ISP): fusion consumes a class
    vocabulary, a shared label level, a modality tag, and a probability matrix,
    and nothing else. :class:`~tulip.pipeline.classifier.DialectClassifier`
    satisfies it structurally, so no import of the concrete class is required to
    depend on this behaviour.
    """

    classes_: tuple[str, ...]
    target: LabelLevel
    task: TaskType

    def predict_proba(self, raws: Sequence[Any]) -> np.ndarray:
        """Return the probability matrix for ``raws``, columns aligned to ``classes_``."""
        ...


@runtime_checkable
class FusionStrategy(Protocol):
    """Combines per-modality probabilities into one distribution per sample.

    ``stack`` has shape ``(n_modalities, n_samples, n_classes)`` -- every
    modality's probabilities already aligned to the same class columns -- and
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

    def fuse(self, stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fuse ``stack`` under ``mask`` into one distribution per sample."""
        ...


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

    Concrete strategies implement only :meth:`_pool` -- how the *present*
    modalities combine into an unnormalised score. This template validates the
    shapes, rejects a sample with no present modality, forces a lone present
    modality to pass through unchanged, and renormalises every row. Centralising
    the contract here is exactly what makes the three strategies provably
    substitutable (LSP), rather than each re-deriving -- and risking diverging
    from -- the same postcondition.
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
        postcondition for *every* strategy -- including log pooling, whose
        single-expert pool ``p ** w`` would otherwise distort a weighted expert.
        """
        singletons = np.flatnonzero(present_counts == 1)
        if singletons.size:
            modality = np.argmax(mask[:, singletons], axis=0)
            combined[singletons] = stack[modality, singletons]

    @staticmethod
    def _renormalise(combined: np.ndarray) -> np.ndarray:
        """Scale every row to sum to 1.

        Upstream invariants keep row sums strictly positive -- each present
        modality's row sums to 1 and no pooling zeroes every class at once -- so
        this never divides by zero or yields ``NaN``.
        """
        return combined / combined.sum(axis=1, keepdims=True)

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
class LogarithmicPoolingFusion(_WeightedFusion):
    """Fuse by the weighted geometric mean -- the log-linear opinion pool.

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


#: OCP factory: a new named strategy registers here without editing any consumer.
_STRATEGY_REGISTRY: dict[str, Callable[[Mapping[str, Any]], FusionStrategy]] = {
    WeightedAverageFusion.kind: lambda params: WeightedAverageFusion(tuple(params["weights"])),
    MaximumFusion.kind: lambda _params: MaximumFusion(),
    LogarithmicPoolingFusion.kind: lambda params: LogarithmicPoolingFusion(
        tuple(params["weights"])
    ),
}


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


class MultimodalClassifier:
    """Fuse a text and an audio classifier into one sample-level predictor.

    This is *not* a :class:`~tulip.pipeline.classifier.DialectClassifier`
    subclass (see the module docstring for the LSP reasoning): it composes two
    :class:`ProbabilisticClassifier` bases and exposes the narrow
    :class:`~tulip.pipeline.protocols.SamplePredictor` contract, so evaluation,
    the CLI, and visualisation treat it interchangeably with the plain and
    hierarchical classifiers.

    The two bases may know different class sets; predictions are aligned to the
    sorted union of their ``classes_``, with a zero column wherever a modality
    never saw a class (a modality cannot vote for a class it does not know).

    Args:
        text: The text-modality base; must have ``task == TaskType.TEXT``.
        audio: The audio-modality base; must have ``task == TaskType.AUDIO``.
        strategy: How to combine the two distributions; defaults to an equal
            :class:`WeightedAverageFusion`.

    Raises:
        ConfigurationError: if the bases disagree on ``target`` level, or either
            base carries the wrong modality ``task``.
    """

    def __init__(
        self,
        *,
        text: ProbabilisticClassifier,
        audio: ProbabilisticClassifier,
        strategy: FusionStrategy | None = None,
    ) -> None:
        if text.target != audio.target:
            raise ConfigurationError(
                "text and audio classifiers must share a target level; got "
                f"text={text.target!r}, audio={audio.target!r}"
            )
        if text.task is not TaskType.TEXT:
            raise ConfigurationError(
                f"the text base must have task={TaskType.TEXT.value!r}, got {text.task!r}"
            )
        if audio.task is not TaskType.AUDIO:
            raise ConfigurationError(
                f"the audio base must have task={TaskType.AUDIO.value!r}, got {audio.task!r}"
            )
        self.text = text
        self.audio = audio
        self.strategy: FusionStrategy = (
            strategy if strategy is not None else WeightedAverageFusion((0.5, 0.5))
        )
        self.target: LabelLevel = text.target
        self.classes_: tuple[str, ...] = self._union_classes()

    # ------------------------------------------------------------------ fit

    def fit(self, samples: Sequence[Sample]) -> Self:
        """Fit each trainable base on the shared sample stream, then realign classes.

        Only bases exposing a callable ``fit`` (i.e. real
        :class:`DialectClassifier`s) are trained; injected probability stubs are
        left as they are, so a stub audio expert stays fixed while a real text
        expert learns. Each base reads its own modality from ``samples``.
        """
        for base in (self.text, self.audio):
            fit = getattr(base, "fit", None)
            if callable(fit):
                fit(samples)
        self.classes_ = self._union_classes()
        return self

    # -------------------------------------------------------------- predict

    def predict_proba_samples(self, samples: Sequence[Sample]) -> np.ndarray:
        """Return the fused probability matrix, columns aligned to :attr:`classes_`.

        Every row sums to 1 and columns follow the sorted union of both bases'
        classes. Per-sample modality availability (text and/or ``audio_path``)
        drives the fusion mask.

        Raises:
            DataError: if any sample provides neither text nor audio.
        """
        samples = list(samples)
        union = self._union_classes()
        self.classes_ = union
        if not samples:
            return np.zeros((0, len(union)), dtype=np.float64)

        texts = [sample.text for sample in samples]
        audios = [sample.audio_path for sample in samples]
        text_present = [text is not None for text in texts]
        audio_present = [audio is not None for audio in audios]
        missing = [
            sample.id
            for sample, has_text, has_audio in zip(
                samples, text_present, audio_present, strict=True
            )
            if not has_text and not has_audio
        ]
        if missing:
            raise DataError(
                f"{len(missing)} sample(s) carry neither text nor audio and cannot be "
                f"fused (first: {missing[0]!r})"
            )

        n_classes = len(union)
        text_matrix = self._modality_matrix(self.text, texts, text_present, union, n_classes)
        audio_matrix = self._modality_matrix(self.audio, audios, audio_present, union, n_classes)
        stack = np.stack([text_matrix, audio_matrix], axis=0)
        mask = np.array([text_present, audio_present], dtype=bool)
        return self.strategy.fuse(stack, mask)

    def predict_samples(self, samples: Sequence[Sample]) -> list[Prediction]:
        """Fuse both modalities and return one :class:`Prediction` per sample, in order.

        Satisfies :class:`~tulip.pipeline.protocols.SamplePredictor`. Every
        prediction carries ``level == self.target`` (the shared level of both
        bases) and the fused distribution over the class union.

        Raises:
            DataError: if any sample provides neither text nor audio.
        """
        probabilities = self.predict_proba_samples(samples)
        predictions: list[Prediction] = []
        for row in probabilities:
            ranked = tuple(
                ClassProbability(label=label, probability=float(p))
                for label, p in zip(self.classes_, row, strict=True)
            )
            predictions.append(
                Prediction(
                    label=self.classes_[int(np.argmax(row))],
                    level=self.target,
                    probabilities=ranked,
                )
            )
        return predictions

    def _union_classes(self) -> tuple[str, ...]:
        """The sorted union of both bases' class vocabularies."""
        return tuple(sorted(set(self.text.classes_) | set(self.audio.classes_)))

    @staticmethod
    def _modality_matrix(
        base: ProbabilisticClassifier,
        raws: list[Any],
        present: list[bool],
        union: tuple[str, ...],
        n_classes: int,
    ) -> np.ndarray:
        """Align one base's probabilities to the shared class union.

        The base scores only the samples that carry its modality; its columns
        are scattered into the union, leaving 0 for any class the base never saw
        and for any absent row (the fusion mask excludes those rows anyway).
        """
        matrix = np.zeros((len(present), n_classes), dtype=np.float64)
        present_idx = [i for i, is_present in enumerate(present) if is_present]
        if not present_idx:
            return matrix
        proba = np.asarray(base.predict_proba([raws[i] for i in present_idx]), dtype=np.float64)
        col_of = {label: index for index, label in enumerate(union)}
        columns = [col_of[label] for label in base.classes_]
        if proba.shape != (len(present_idx), len(columns)):
            raise DataError(
                f"{type(base).__name__} returned probabilities of shape {proba.shape}, "
                f"expected {(len(present_idx), len(columns))}"
            )
        matrix[np.ix_(present_idx, columns)] = proba
        return matrix

    # -------------------------------------------------------------- persist

    #: Sub-directory names and sidecar file within a saved artifact.
    _TEXT_DIR: ClassVar[str] = "text"
    _AUDIO_DIR: ClassVar[str] = "audio"
    _SIDECAR: ClassVar[str] = "fusion.json"

    def save(self, path: Path | str) -> Path:
        """Persist both base classifiers and the fusion strategy under one directory.

        Layout::

            <path>/text/        the text base's DialectClassifier artifact
            <path>/audio/       the audio base's DialectClassifier artifact
            <path>/fusion.json  strategy kind/params and the shared target level

        Raises:
            ConfigurationError: if a base cannot be persisted (only
                :class:`DialectClassifier` bases expose ``save``; probability
                stubs do not) or the strategy cannot be serialised.
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self._save_base(self.text, target / self._TEXT_DIR)
        self._save_base(self.audio, target / self._AUDIO_DIR)
        sidecar = {
            "kind": "MultimodalClassifier",
            "target": self.target.value,
            "strategy": self._strategy_config(self.strategy),
        }
        payload = json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True)
        (target / self._SIDECAR).write_text(payload + "\n", encoding="utf-8", newline="\n")
        _logger.info("saved MultimodalClassifier to %s", target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Restore a classifier saved by :meth:`save`, ready to predict.

        Both bases are reloaded as :class:`DialectClassifier`s and the strategy
        is rebuilt from its recorded kind/params via :func:`build_strategy`.

        Raises:
            DataError: if the artifact is missing, incomplete, or was not
                written by :meth:`save`.
        """
        source = Path(path)
        sidecar_path = source / cls._SIDECAR
        if not sidecar_path.is_file():
            raise DataError(f"no MultimodalClassifier artifact at {source}: missing {cls._SIDECAR}")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataError(f"corrupt fusion sidecar at {sidecar_path}: {exc}") from exc
        if not isinstance(sidecar, dict):
            raise DataError(f"corrupt fusion sidecar at {sidecar_path}: expected a JSON object")
        if sidecar.get("kind") != "MultimodalClassifier":
            raise DataError(
                f"artifact at {source} was not saved by MultimodalClassifier.save() "
                f"(kind={sidecar.get('kind')!r})"
            )
        text = DialectClassifier.load(source / cls._TEXT_DIR)
        audio = DialectClassifier.load(source / cls._AUDIO_DIR)
        strategy_cfg = sidecar.get("strategy", {})
        strategy = build_strategy(
            strategy_cfg.get("kind", WeightedAverageFusion.kind),
            strategy_cfg.get("params", {}),
        )
        return cls(text=text, audio=audio, strategy=strategy)

    @staticmethod
    def _save_base(base: ProbabilisticClassifier, path: Path) -> None:
        """Persist one base, requiring it to expose a ``save`` (DialectClassifier does)."""
        save = getattr(base, "save", None)
        if not callable(save):
            raise ConfigurationError(
                f"base classifier {type(base).__name__} cannot be persisted; it exposes no "
                "'save' (only DialectClassifier bases are persistable)"
            )
        save(path)

    @staticmethod
    def _strategy_config(strategy: FusionStrategy) -> dict[str, Any]:
        """Serialise a strategy to ``{"kind", "params"}`` for the sidecar.

        Keeps :class:`FusionStrategy` narrow (ISP: ``fuse`` only) by discovering
        ``kind``/``config`` structurally; a strategy lacking them is not
        persistable, which is a configuration error rather than a silent gap.
        """
        kind = getattr(strategy, "kind", None)
        config = getattr(strategy, "config", None)
        if not isinstance(kind, str) or not callable(config):
            raise ConfigurationError(
                f"fusion strategy {type(strategy).__name__} cannot be serialised; it exposes "
                "no 'kind'/'config' (only the built-in strategies are persistable)"
            )
        return {"kind": kind, "params": config()}

    def __repr__(self) -> str:
        return (
            f"MultimodalClassifier(text={self.text!r}, audio={self.audio!r}, "
            f"strategy={type(self.strategy).__name__}, target={self.target.value})"
        )


__all__ = [
    "FusionStrategy",
    "LogarithmicPoolingFusion",
    "MaximumFusion",
    "MultimodalClassifier",
    "ProbabilisticClassifier",
    "WeightedAverageFusion",
    "build_strategy",
]
