"""The user-facing dialect classification facade.

:class:`DialectClassifier` composes registry-configured feature extractors
and a model into one trainable object working directly on
:class:`~tulip.core.types.Sample` streams, and returns rich
:class:`~tulip.core.types.Prediction` objects (full probability distribution,
top-k, optional abstention) instead of bare labels. Explanations delegate to
:mod:`tulip.explain`, persistence to :mod:`tulip.models.persistence`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.pipeline import Pipeline

from tulip.config.schemas import ComponentConfig
from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import Explanation, Prediction, Sample, TaskType
from tulip.labels.taxonomy import LabelLevel
from tulip.models import MODELS
from tulip.models.persistence import load_model, save_model
from tulip.pipeline._assembly import (
    predictions_from_proba,
    raw_input_of,
    raws_for_task,
    validate_abstain_threshold,
)
from tulip.pipeline.explaining import PredictionExplainer
from tulip.utils.logging import get_logger
from tulip.utils.seed import set_global_seed

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

_logger = get_logger(__name__)

ComponentLike = ComponentConfig | Mapping[str, Any] | str


def _coerce_component(entry: ComponentLike) -> ComponentConfig:
    """Normalise a component reference (config object, mapping, or bare name)."""
    if isinstance(entry, ComponentConfig):
        return entry
    if isinstance(entry, str):
        return ComponentConfig(name=entry)
    return ComponentConfig.model_validate(dict(entry))


@dataclass(frozen=True)
class LabelledBatch:
    """Raw model inputs paired with labels at one granularity.

    Produced by :meth:`DialectClassifier.labelled_batch`; consumed by training
    and by evaluation (:func:`tulip.pipeline.experiment.evaluate_samples`).

    ``samples`` holds the surviving :class:`~tulip.core.types.Sample` objects
    positionally aligned with ``raws``/``labels`` (skipped samples are absent).
    Training ignores it; per-sample evaluation
    (:func:`tulip.pipeline.experiment.collect_predictions`) uses it to attach
    each record's slice keys (id, source, speaker, length) without re-filtering.
    """

    raws: list[Any]
    labels: list[str]
    n_skipped: int
    samples: list[Sample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.raws)


class DialectClassifier:
    """Train, predict, explain, and persist a dialect classification pipeline.

    Two composition shapes are supported, chosen by ``features``:

    * **Feature pipeline** (classical): a non-empty ``features`` list builds a
      :class:`sklearn.pipeline.FeatureUnion` (text or audio registry) feeding
      the configured model.
    * **Raw-input model** (neural): an empty ``features`` list passes raw
      texts / audio paths straight to the model (transformer and speech
      wrappers tokenise/decode internally).

    Args:
        model: Model reference (registry name, mapping, or ComponentConfig).
        features: Feature component references; empty for raw-input models.
        task: Input modality; decides which sample field and feature registry
            are used.
        target: Label granularity to train against (see LabelLevel).
        abstain_threshold: When set, predictions whose top probability falls
            below it abstain (``label=None``) instead of guessing;
            uncertainty-aware classification for out-of-scope inputs.
        seed: Seed applied before fitting (numpy/random/torch when present).
    """

    def __init__(
        self,
        *,
        model: ComponentLike,
        features: Sequence[ComponentLike] = (),
        task: TaskType = TaskType.TEXT,
        target: LabelLevel = LabelLevel.DIALECT,
        abstain_threshold: float | None = None,
        seed: int = 42,
    ) -> None:
        validate_abstain_threshold(abstain_threshold)
        self.model_config = _coerce_component(model)
        self.feature_configs = tuple(_coerce_component(entry) for entry in features)
        self.task = TaskType(task)
        self.target = LabelLevel(target)
        self.abstain_threshold = abstain_threshold
        self.seed = seed

        self.pipeline_: Any | None = None
        self.classes_: tuple[str, ...] = ()
        self._train_samples: list[Sample] = []
        self._prediction_explainer: PredictionExplainer | None = None

    # ------------------------------------------------------------------ fit

    def fit(self, samples: Sequence[Sample]) -> Self:
        """Train on the samples that carry both the input modality and the label.

        Samples missing text/audio for the configured task, or missing a label
        at the target level, are skipped with a logged count: corpora are
        heterogeneous and partial labelling is the norm, but training on
        nothing is an error.

        Raises:
            DataError: if no trainable samples remain.
        """
        batch = self.labelled_batch(samples)
        if not batch.raws:
            raise DataError(
                f"none of the {len(samples)} samples are trainable for task={self.task.value} "
                f"at level={self.target.value} (skipped: {batch.n_skipped})"
            )
        if batch.n_skipped:
            _logger.info(
                "training on %d samples (skipped %d without %s input or %r label)",
                len(batch),
                batch.n_skipped,
                self.task.value,
                self.target.value,
            )
        set_global_seed(self.seed)
        try:
            estimator = MODELS.create(self.model_config.name, **self.model_config.params)
        except (TypeError, ValueError) as exc:
            # A mistyped or unsupported params entry reaches the estimator
            # constructor as an unexpected keyword; report the config, not a traceback.
            raise ConfigurationError(
                f"cannot build model {self.model_config.name!r} from its configured "
                f"params {dict(self.model_config.params)!r}: {exc}"
            ) from exc
        self._require_raw_capable_model()
        if self.feature_configs:
            self.pipeline_ = Pipeline([("features", self._build_features()), ("model", estimator)])
        else:
            self.pipeline_ = estimator
        self.pipeline_.fit(batch.raws, batch.labels)
        self.classes_ = tuple(str(label) for label in np.asarray(self.pipeline_.classes_))
        self._train_samples = [sample for sample in samples if self._raw_of(sample) is not None]
        self._prediction_explainer = None  # explainer state is rebuilt lazily on demand
        return self

    def _require_raw_capable_model(self) -> None:
        """Reject an empty feature list unless the model consumes raw input itself.

        Without this, a classical estimator is handed raw strings (or audio
        paths) and dies deep inside scikit-learn with ``could not convert string
        to float``, which tells the user nothing. Raw-capable models declare
        themselves with ``metadata={"raw_input": True}`` at registration, which
        is the capability mechanism docs/architecture.md prescribes; consumers
        query the registry rather than hardcoding a list of model names.

        Raises:
            ConfigurationError: if ``features`` is empty and the model needs them.
        """
        if self.feature_configs:
            return
        if MODELS.metadata(self.model_config.name).get("raw_input", False):
            return
        raw_capable = sorted(
            name for name in MODELS.names() if MODELS.metadata(name).get("raw_input", False)
        )
        raise ConfigurationError(
            f"model {self.model_config.name!r} cannot consume raw {self.task.value} input, "
            f"so at least one feature extractor is required. Either pass features "
            f"(e.g. char_tfidf), or choose a raw-input model: {', '.join(raw_capable)}"
        )

    def _build_features(self) -> Any:
        """Build the feature union for the configured task's registry."""
        if self.task is TaskType.TEXT:
            from tulip.features.text.composite import build_text_features

            return build_text_features(list(self.feature_configs))
        from tulip.features.audio.composite import build_audio_features

        return build_audio_features(list(self.feature_configs))

    def _raw_of(self, sample: Sample) -> Any | None:
        """Extract the raw model input for one sample, or ``None`` if absent."""
        return raw_input_of(sample, self.task)

    def labelled_batch(self, samples: Sequence[Sample]) -> LabelledBatch:
        """Pair each sample's raw input with its label at the target level.

        Samples missing the task's input modality or the target-level label
        are skipped and counted, not errors: corpora are heterogeneous and
        partial labelling is the norm. Callers (training, evaluation) decide
        whether an empty batch is acceptable.
        """
        raws: list[Any] = []
        labels: list[str] = []
        kept: list[Sample] = []
        skipped = 0
        for sample in samples:
            raw = self._raw_of(sample)
            label = sample.labels.at_level(self.target)
            if raw is None or label is None:
                skipped += 1
                continue
            raws.append(raw)
            labels.append(str(label))
            kept.append(sample)
        return LabelledBatch(raws=raws, labels=labels, n_skipped=skipped, samples=kept)

    # -------------------------------------------------------------- predict

    def predict(self, raw: Any) -> Prediction:
        """Classify one raw input (text or audio path)."""
        return self.predict_batch([raw])[0]

    def predict_samples(self, samples: Sequence[Sample]) -> list[Prediction]:
        """Classify samples, reading whichever modality this classifier was built for.

        The adapter that satisfies
        :class:`~tulip.pipeline.protocols.SamplePredictor`, so a plain
        classifier, a hierarchical one, and a multimodal one are
        interchangeable to their consumers.

        Unlike :meth:`labelled_batch`, a missing modality is an error rather
        than a skip: prediction has no labels to fall back on, and silently
        dropping rows would misalign the caller's own bookkeeping.

        Raises:
            DataError: if any sample lacks this classifier's input modality.
        """
        return self.predict_batch(raws_for_task(samples, self.task))

    def predict_batch(self, raws: Sequence[Any]) -> list[Prediction]:
        """Classify a batch of raw inputs, one :class:`Prediction` each.

        Assembly is shared with the calibrated and multimodal classifiers via
        :func:`~tulip.pipeline._assembly.predictions_from_proba` (see there for
        the ~40% Prediction-construction cost note); bulk consumers that only
        need the probability matrix should call :meth:`predict_proba` directly.
        """
        probabilities = self.predict_proba(raws)
        return predictions_from_proba(
            probabilities, self.classes_, self.target, abstain_threshold=self.abstain_threshold
        )

    def predict_proba(self, raws: Sequence[Any]) -> np.ndarray:
        """Return the probability matrix for a batch, columns aligned with ``classes_``.

        Models without native probabilities degrade to one-hot rows built from
        their hard predictions (with a logged warning), so downstream code can
        rely on this method existing, the same guarantee the
        :class:`~tulip.core.interfaces.Classifier` protocol makes.

        Raises:
            ConfigurationError: if the classifier is not fitted.
        """
        self._require_fitted()
        # Type narrowing for mypy, not a runtime check: _require_fitted() has
        # already raised if the pipeline is None, so stripping asserts under
        # `python -O` cannot turn this into a silent failure.
        assert self.pipeline_ is not None  # noqa: S101
        inputs = list(raws)
        if hasattr(self.pipeline_, "predict_proba"):
            return np.asarray(self.pipeline_.predict_proba(inputs), dtype=np.float64)
        _logger.warning(
            "%s has no predict_proba; probabilities degrade to one-hot predictions",
            type(self.pipeline_).__name__,
        )
        predicted = np.asarray(self.pipeline_.predict(inputs)).astype(str)
        matrix = np.zeros((len(inputs), len(self.classes_)), dtype=np.float64)
        index_of = {label: i for i, label in enumerate(self.classes_)}
        for row, label in enumerate(predicted):
            matrix[row, index_of[label]] = 1.0
        return matrix

    # -------------------------------------------------------------- explain

    def explain(self, raw: Any, method: str = "top_tfidf", **kwargs: Any) -> Explanation:
        """Explain one prediction with the requested explainer.

        Args:
            raw: The raw input to explain.
            method: Explainer registry name (``top_tfidf``, ``lime``,
                ``shap``, ``attention``, ``nearest_examples``).
            **kwargs: Forwarded to the explainer's ``explain`` call.

        Raises:
            ConfigurationError: if the classifier is unfitted, or the method
                is incompatible with the composed pipeline.
        """
        self._require_fitted()
        if self._prediction_explainer is None:
            self._prediction_explainer = PredictionExplainer(
                pipeline=self.pipeline_,
                task=self.task,
                train_samples=self._train_samples,
            )
        return self._prediction_explainer.explain(raw, method=method, **kwargs)

    # -------------------------------------------------------------- persist

    def save(self, path: Path | str) -> Path:
        """Persist the fitted pipeline plus the full classifier configuration.

        The in-memory training samples are *not* serialised (artifacts stay
        small and corpora may be unredistributable); ``nearest_examples``
        therefore requires a live, fitted classifier.
        """
        self._require_fitted()
        return save_model(
            self.pipeline_,
            path,
            metadata={
                "kind": "DialectClassifier",
                "task": self.task.value,
                "target": self.target.value,
                "abstain_threshold": self.abstain_threshold,
                "seed": self.seed,
                "features": [c.model_dump(mode="json") for c in self.feature_configs],
                "model": self.model_config.model_dump(mode="json"),
            },
        )

    @classmethod
    def load(cls, path: Path | str) -> DialectClassifier:
        """Restore a classifier saved by :meth:`save`, ready to predict.

        Raises:
            DataError: if the artifact is missing/corrupt or was not written
                by :meth:`save`.
        """
        pipeline, sidecar = load_model(path)
        stored = sidecar.get("metadata", {})
        if stored.get("kind") != "DialectClassifier":
            raise DataError(
                f"artifact at {path} was not saved by DialectClassifier.save() "
                f"(kind={stored.get('kind')!r})"
            )
        from pydantic import ValidationError

        try:
            classifier = cls(
                model=stored["model"],
                features=stored.get("features", ()),
                task=TaskType(stored.get("task", TaskType.TEXT.value)),
                target=LabelLevel(stored.get("target", LabelLevel.DIALECT.value)),
                abstain_threshold=stored.get("abstain_threshold"),
                seed=int(stored.get("seed", 42)),
            )
        except (KeyError, ValueError, TypeError, ValidationError) as exc:
            raise DataError(f"corrupt DialectClassifier artifact at {path}: {exc}") from exc
        classifier.pipeline_ = pipeline
        classes = sidecar.get("classes") or np.asarray(pipeline.classes_).tolist()
        classifier.classes_ = tuple(str(label) for label in classes)
        return classifier

    def _require_fitted(self) -> None:
        if self.pipeline_ is None:
            raise ConfigurationError("this DialectClassifier is not fitted yet; call fit() first")

    def __repr__(self) -> str:
        features = [c.name for c in self.feature_configs] or "raw-input"
        return (
            f"DialectClassifier(model={self.model_config.name!r}, features={features}, "
            f"task={self.task.value}, target={self.target.value})"
        )


__all__ = ["DialectClassifier", "LabelledBatch"]
