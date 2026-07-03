"""Explanation routing for fitted classification pipelines.

Split out of :class:`~tulip.pipeline.classifier.DialectClassifier` so the
facade stays focused on train/predict/persist, while the knowledge of *which
explainer sees which object* lives in one place:

* ``top_tfidf`` / ``lime`` / ``shap`` inspect the full prediction pipeline;
* ``attention`` needs the bare transformer wrapper (its ``model_`` and
  ``tokenizer_`` attributes), not an sklearn feature pipeline;
* ``nearest_examples`` needs a similarity index over the training samples,
  built lazily here from the pipeline's own feature transformer (or, for
  raw-input models, from a dedicated character TF-IDF retrieval vectorizer).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.pipeline import Pipeline

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import Explanation, Sample, TaskType
from tulip.explain import get_explainer
from tulip.explain.neighbors import NearestExamplesExplainer
from tulip.utils.logging import get_logger

_logger = get_logger(__name__)

__all__ = ["PredictionExplainer"]


class PredictionExplainer:
    """Routes explanation requests to the right explainer and target object.

    Instances are cheap to construct; the nearest-example index (the only
    expensive state) is built on first use and cached for the lifetime of
    this object.

    Args:
        pipeline: The fitted prediction object — an sklearn
            :class:`~sklearn.pipeline.Pipeline` (features + model) or a
            raw-input model taking texts/audio paths directly.
        task: Input modality of the pipeline.
        train_samples: The samples the pipeline was fitted on. May be empty
            (e.g. for a model restored from disk), in which case
            ``nearest_examples`` is unavailable.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        task: TaskType,
        train_samples: Sequence[Sample] = (),
    ) -> None:
        self._pipeline = pipeline
        self._task = TaskType(task)
        self._train_samples = list(train_samples)
        self._neighbor_explainer: NearestExamplesExplainer | None = None
        self._retrieval_vectorizer: Any | None = None

    def explain(self, raw: Any, method: str = "top_tfidf", **kwargs: Any) -> Explanation:
        """Explain one prediction with the requested explainer.

        Args:
            raw: The raw input to explain.
            method: Explainer registry name (``top_tfidf``, ``lime``,
                ``shap``, ``attention``, ``nearest_examples``).
            **kwargs: Forwarded to the explainer's ``explain`` call.

        Raises:
            ConfigurationError: if the method is incompatible with the
                pipeline's composition (e.g. ``nearest_examples`` without
                training samples).
            UnknownComponentError: if ``method`` is not a registered explainer.
        """
        if method == "nearest_examples":
            return self._neighbor_index().explain(self._pipeline, raw, **kwargs)
        if method == "attention":
            # Attention lives on the transformer wrapper itself, not on an
            # sklearn feature pipeline.
            return get_explainer(method).explain(self._final_estimator(), raw, **kwargs)
        return get_explainer(method).explain(self._pipeline, raw, **kwargs)

    def _final_estimator(self) -> Any:
        """The model at the end of the pipeline (or the raw-input model itself)."""
        if isinstance(self._pipeline, Pipeline):
            return self._pipeline.steps[-1][1]
        return self._pipeline

    def _neighbor_index(self) -> NearestExamplesExplainer:
        """Build (once) and return the nearest-examples explainer.

        Depends on the concrete class rather than the ``Explainer`` protocol
        because this is exactly the site that needs its extra capability —
        building the index — which the protocol deliberately does not carry.
        """
        if self._neighbor_explainer is not None:
            return self._neighbor_explainer
        if not self._train_samples:
            raise ConfigurationError(
                "nearest_examples needs the in-memory training samples; a classifier "
                "restored with DialectClassifier.load() must be refitted (or use another "
                "explainer)"
            )
        explainer = NearestExamplesExplainer()
        explainer.index(self._train_samples, self._retrieval_transformer())
        self._neighbor_explainer = explainer
        return explainer

    def _retrieval_transformer(self) -> Any:
        """A fitted text transformer for similarity retrieval.

        Uses the pipeline's own feature union when present; raw-input models
        get a dedicated character TF-IDF fitted on the training texts, so
        nearest-example retrieval works for transformer models too.
        """
        if isinstance(self._pipeline, Pipeline):
            return self._pipeline[:-1]
        if self._task is not TaskType.TEXT:
            raise ConfigurationError(
                "nearest_examples retrieval is text-based; audio pipelines without a "
                "feature union are not supported"
            )
        if self._retrieval_vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            texts = [sample.text or "" for sample in self._train_samples]
            _logger.debug("fitting retrieval TF-IDF over %d training texts", len(texts))
            self._retrieval_vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
            ).fit(texts)
        return self._retrieval_vectorizer
