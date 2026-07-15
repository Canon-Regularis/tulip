"""LIME text explanations for any probabilistic tulip pipeline.

LIME fits a sparse local linear surrogate around the input by perturbing it
(removing words) and observing how ``predict_proba`` responds. It is
model-agnostic: the only requirement on the pipeline is ``predict_proba``
over raw texts. This makes it the right tool for non-linear models where
``top_tfidf`` refuses to run.

The ``lime`` package is an optional dependency (extra ``explain``) imported
lazily inside :meth:`LimeExplainer.explain`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import Explanation, TokenAttribution
from tulip.explain._shared import as_text, predicted_class
from tulip.explain.registry import EXPLAINERS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

logger = get_logger(__name__)

__all__ = ["LimeExplainer"]


@EXPLAINERS.register("lime", metadata={"extra": "explain"})
class LimeExplainer:
    """Local surrogate (LIME) word attributions for the predicted class.

    Deterministic for a fixed ``seed``: the seed drives LIME's perturbation
    sampling, so repeated calls on the same input produce identical
    attributions.

    Attributes:
        num_features: Maximum number of words in the explanation.
        num_samples: Number of perturbed texts used to fit the surrogate
            (more is more stable but proportionally slower).
        seed: Seed for the perturbation sampler.
    """

    def __init__(self, num_features: int = 10, num_samples: int = 1000, seed: int = 42) -> None:
        """Configure the explainer.

        Args:
            num_features: Maximum number of attributed words to return.
            num_samples: Perturbation sample count for the local surrogate.
            seed: Seed making the perturbation sampling reproducible.

        Raises:
            ConfigurationError: if a parameter is out of range.
        """
        if num_features < 1:
            raise ConfigurationError(f"num_features must be >= 1, got {num_features}")
        if num_samples < 2:
            raise ConfigurationError(f"num_samples must be >= 2, got {num_samples}")
        self.num_features = num_features
        self.num_samples = num_samples
        self.seed = seed

    def explain(self, pipeline: Any, raw_input: Any, **kwargs: Any) -> Explanation:
        """Explain one prediction with a LIME local surrogate.

        Args:
            pipeline: A fitted classifier exposing ``predict_proba`` over raw
                texts and ``classes_``.
            raw_input: The raw document to explain.
            **kwargs: ``num_features``, ``num_samples``, and ``seed`` override
                the constructor values.

        Returns:
            An :class:`Explanation` with signed word attributions towards the
            predicted class.

        Raises:
            MissingDependencyError: if ``lime`` is not installed.
            ConfigurationError: if the pipeline lacks ``predict_proba``.
        """
        lime_text = optional_import(
            "lime.lime_text", extra="explain", purpose="LIME text explanations"
        )
        num_features = int(kwargs.get("num_features", self.num_features))
        num_samples = int(kwargs.get("num_samples", self.num_samples))
        seed = int(kwargs.get("seed", self.seed))
        text = as_text(raw_input)
        class_index, predicted_label, class_names = predicted_class(pipeline, text)

        def probability_fn(texts: Any) -> np.ndarray:
            return np.asarray(pipeline.predict_proba(list(texts)), dtype=np.float64)

        explainer = lime_text.LimeTextExplainer(class_names=class_names, random_state=seed)
        logger.debug(
            "running LIME with %d samples on %d-class model", num_samples, len(class_names)
        )
        lime_result = explainer.explain_instance(
            text,
            probability_fn,
            labels=(class_index,),
            num_features=num_features,
            num_samples=num_samples,
        )
        attributions = tuple(
            TokenAttribution(token=str(token), weight=float(weight))
            for token, weight in lime_result.as_list(label=class_index)
        )
        # lime stores the surrogate R^2 as a scalar or a per-label dict
        # depending on version; normalise to a plain float or None.
        score: Any = getattr(lime_result, "score", None)
        if isinstance(score, dict):
            score = score.get(class_index)
        surrogate_score = float(score) if isinstance(score, (int, float, np.floating)) else None
        return Explanation(
            method="lime",
            predicted_label=predicted_label,
            attributions=attributions,
            details={
                "class_names": class_names,
                "num_samples": num_samples,
                "num_features": num_features,
                "seed": seed,
                "surrogate_score": surrogate_score,
            },
        )
