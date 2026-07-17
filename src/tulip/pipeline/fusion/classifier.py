"""The multimodal classifier that composes a text and an audio base by late fusion.

A :class:`~tulip.pipeline.classifier.DialectClassifier` fixes ``task`` to
TEXT *xor* AUDIO, so the two modalities a :class:`~tulip.core.types.Sample`
carries (``text`` and ``audio_path``) never combine, even though many corpora
provide both. :class:`MultimodalClassifier` fuses them at the probability level.

**Why not extend the task enum.** ``TaskType`` (in the frozen ``core/types``)
has only ``TEXT`` and ``AUDIO``; adding a ``MULTIMODAL`` member would be the
wrong design even if the enum were editable. A multimodal model is not a third
*modality*; it is a *composition* of two single-modality models, each an
expert on its own input. Modelling it as two :class:`DialectClassifier`s joined
by a :class:`~tulip.pipeline.fusion.strategies.FusionStrategy` keeps each base
independently trainable, swappable, and persistable.

**Why this is not a DialectClassifier subclass (LSP).**
``DialectClassifier.predict_batch`` guarantees every returned
:class:`~tulip.core.types.Prediction` has ``level == self.target`` *and* that its
inputs are raw values of one modality. A multimodal classifier reads *both*
modalities from a whole :class:`Sample`, so it cannot honour the second
postcondition and must not be substitutable for a ``DialectClassifier``. It
relates to its bases by *composition* and exposes only the narrow
:class:`~tulip.pipeline.protocols.SamplePredictor` contract.

**Why the dependency is a structural type (DIP).** The classifier depends on
:class:`~tulip.pipeline.protocols.ProbabilisticClassifier` (just ``classes_``,
``target``, ``task``, and ``predict_proba``), not on concrete
``DialectClassifier``. Any object exposing those satisfies it, which lets tests
inject a cheap deterministic audio stub instead of training a real speech model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from tulip._jsonio import read_json_object
from tulip._serialize import write_sorted_json
from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import Prediction, TaskType
from tulip.pipeline._assembly import predictions_from_proba
from tulip.pipeline.classifier import DialectClassifier
from tulip.pipeline.fusion.strategies import (
    FusionStrategy,
    WeightedAverageFusion,
    build_strategy,
)
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Self

    from tulip.core.types import Sample
    from tulip.labels.taxonomy import LabelLevel
    from tulip.pipeline.protocols import ProbabilisticClassifier

_logger = get_logger(__name__)

__all__ = ["MultimodalClassifier"]


class MultimodalClassifier:
    """Fuse a text and an audio classifier into one sample-level predictor.

    This is *not* a :class:`~tulip.pipeline.classifier.DialectClassifier`
    subclass (see the module docstring for the LSP reasoning): it composes two
    :class:`~tulip.pipeline.protocols.ProbabilisticClassifier` bases and exposes
    the narrow :class:`~tulip.pipeline.protocols.SamplePredictor` contract, so
    evaluation, the CLI, and visualisation treat it interchangeably with the
    plain and hierarchical classifiers.

    The two bases may know different class sets; predictions are aligned to the
    sorted union of their ``classes_``, with a zero column wherever a modality
    never saw a class (a modality cannot vote for a class it does not know).

    Args:
        text: The text-modality base; must have ``task == TaskType.TEXT``.
        audio: The audio-modality base; must have ``task == TaskType.AUDIO``.
        strategy: How to combine the two distributions; defaults to an equal
            :class:`~tulip.pipeline.fusion.strategies.WeightedAverageFusion`.

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
        bases) and the fused distribution over the class union. Fusion has no
        abstention of its own, so the shared assembly is called with
        ``abstain_threshold=None``.

        Raises:
            DataError: if any sample provides neither text nor audio.
        """
        probabilities = self.predict_proba_samples(samples)
        return predictions_from_proba(probabilities, self.classes_, self.target)

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
    _KIND: ClassVar[str] = "MultimodalClassifier"
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
            "kind": self._KIND,
            "target": self.target.value,
            "strategy": self._strategy_config(self.strategy),
        }
        write_sorted_json(target / self._SIDECAR, sidecar)
        _logger.info("saved MultimodalClassifier to %s", target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Restore a classifier saved by :meth:`save`, ready to predict.

        Both bases are reloaded as :class:`DialectClassifier`s and the strategy
        is rebuilt from its recorded kind/params via
        :func:`~tulip.pipeline.fusion.strategies.build_strategy`.

        Raises:
            DataError: if the artifact is missing, incomplete, or was not
                written by :meth:`save`.
        """
        source = Path(path)
        sidecar_path = source / cls._SIDECAR
        if not sidecar_path.is_file():
            raise DataError(f"no MultimodalClassifier artifact at {source}: missing {cls._SIDECAR}")
        sidecar = read_json_object(sidecar_path, what="fusion sidecar")
        if sidecar.get("kind") != cls._KIND:
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

        Keeps :class:`~tulip.pipeline.fusion.strategies.FusionStrategy` narrow
        (ISP: ``fuse`` only) by discovering ``kind``/``config`` structurally; a
        strategy lacking them is not persistable, which is a configuration error
        rather than a silent gap.
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
