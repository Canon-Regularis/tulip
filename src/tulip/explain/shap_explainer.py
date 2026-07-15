"""SHAP text explanations for any probabilistic tulip pipeline.

Uses ``shap.maskers.Text`` (whitespace/regex tokenisation of the raw input)
with ``shap.Explainer`` over ``pipeline.predict_proba``. With a text masker
shap normally selects the Partition explainer, which is reasonably fast and
exact under its token-clustering assumptions; if shap ever falls back to
``KernelExplainer`` (e.g. an explicitly requested algorithm), runtime grows
with the number of model evaluations, which is why ``max_evals`` caps the
total evaluation budget and long inputs should be truncated by the caller.

The ``shap`` package is an optional dependency (extra ``explain``) imported
lazily inside :meth:`ShapExplainer.explain`.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import Explanation, TokenAttribution
from tulip.explain._shared import as_text, predicted_class
from tulip.explain.registry import EXPLAINERS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

logger = get_logger(__name__)

__all__ = ["ShapExplainer"]


@EXPLAINERS.register("shap", metadata={"extra": "explain"})
class ShapExplainer:
    """SHAP token attributions towards the predicted class.

    Attributes:
        max_evals: Cap on model evaluations per explanation. This bounds both
            runtime and, for sampling-based fallbacks, the effective sample
            size; raise it for long texts (shap requires roughly
            ``2 * n_tokens + 1`` evaluations as a floor).
        top_k: Maximum number of attributions kept (0 keeps every token).
        algorithm: shap algorithm hint (``"auto"`` selects Partition for the
            text masker; ``"permutation"``/``"exact"`` are also valid).
    """

    def __init__(self, max_evals: int = 200, *, top_k: int = 0, algorithm: str = "auto") -> None:
        """Configure the explainer.

        Args:
            max_evals: Evaluation budget per explanation (>= 10).
            top_k: Keep only the ``top_k`` largest-|weight| attributions
                (0 disables the cap).
            algorithm: Forwarded to ``shap.Explainer``.

        Raises:
            ConfigurationError: if a parameter is out of range.
        """
        if max_evals < 10:
            raise ConfigurationError(f"max_evals must be >= 10, got {max_evals}")
        if top_k < 0:
            raise ConfigurationError(f"top_k must be >= 0, got {top_k}")
        self.max_evals = max_evals
        self.top_k = top_k
        self.algorithm = algorithm

    def explain(self, pipeline: Any, raw_input: Any, **kwargs: Any) -> Explanation:
        """Explain one prediction with SHAP values over input tokens.

        Args:
            pipeline: A fitted classifier exposing ``predict_proba`` over raw
                texts and ``classes_``.
            raw_input: The raw document to explain.
            **kwargs: ``max_evals``, ``top_k``, and ``algorithm`` override the
                constructor values.

        Returns:
            An :class:`Explanation` with per-token SHAP values towards the
            predicted class (weights sum to roughly the probability shift
            from the masked baseline).

        Raises:
            MissingDependencyError: if ``shap`` is not installed.
            ConfigurationError: if the pipeline lacks ``predict_proba``.
        """
        shap = optional_import("shap", extra="explain", purpose="SHAP text explanations")
        max_evals = int(kwargs.get("max_evals", self.max_evals))
        top_k = int(kwargs.get("top_k", self.top_k))
        algorithm = str(kwargs.get("algorithm", self.algorithm))
        text = as_text(raw_input)
        class_index, predicted_label, _ = predicted_class(pipeline, text)
        class_names = [str(label) for label in np.asarray(pipeline.classes_)]

        def probability_fn(texts: Any) -> np.ndarray:
            return np.asarray(pipeline.predict_proba([str(t) for t in texts]), dtype=np.float64)

        masker = shap.maskers.Text(r"\W+")
        explainer = shap.Explainer(
            probability_fn, masker, output_names=class_names, algorithm=algorithm
        )
        logger.debug("running SHAP (max_evals=%d) on %d-class model", max_evals, len(class_names))
        shap_result = explainer([text], max_evals=max_evals, silent=True)

        values = np.asarray(shap_result.values[0], dtype=np.float64)
        per_token = values[:, class_index] if values.ndim == 2 else values
        tokens = [str(token) for token in np.asarray(shap_result.data[0], dtype=object)]

        # shap's Text masker keeps trailing separators attached to tokens
        # ("Hej," instead of "Hej"); strip non-word edges so attributions are
        # readable words, dropping tokens that are pure punctuation.
        cleaned = (
            (re.sub(r"^\W+|\W+$", "", token, flags=re.UNICODE), float(weight))
            for token, weight in zip(tokens, per_token, strict=True)
        )
        pairs = [(token, weight) for token, weight in cleaned if token]
        if top_k > 0:
            pairs = sorted(pairs, key=lambda pair: abs(pair[1]), reverse=True)[:top_k]
        attributions = tuple(
            TokenAttribution(token=token, weight=weight) for token, weight in pairs
        )
        base_values = np.asarray(shap_result.base_values, dtype=np.float64).ravel()
        base_value = float(base_values[class_index]) if class_index < base_values.size else None
        return Explanation(
            method="shap",
            predicted_label=predicted_label,
            attributions=attributions,
            details={
                "class_names": class_names,
                "max_evals": max_evals,
                "base_value": base_value,
                "algorithm": self.algorithm,
            },
        )
