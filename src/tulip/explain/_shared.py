"""Internal helpers shared by the explainer implementations.

Kept private (underscore module): the public surface of :mod:`tulip.explain`
is the ``EXPLAINERS`` registry and the explainer classes themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tulip.core.exceptions import ConfigurationError


def as_text(raw_input: Any) -> str:
    """Coerce a raw explainer input to a non-empty string.

    Args:
        raw_input: The raw input passed to ``explain`` (usually a str).

    Returns:
        The input as a string.

    Raises:
        ConfigurationError: if the input is empty or whitespace-only, because
            no token-level explanation can be produced for it.
    """
    text = str(raw_input)
    if not text.strip():
        raise ConfigurationError("cannot explain an empty input text")
    return text


def predicted_class(pipeline: Any, text: str) -> tuple[int, str, np.ndarray]:
    """Return the predicted class for ``text`` under a probabilistic pipeline.

    Args:
        pipeline: A fitted object exposing ``predict_proba`` and ``classes_``.
        text: The raw input document.

    Returns:
        ``(index, label, probabilities)`` where ``index`` is the argmax column,
        ``label`` the corresponding entry of ``classes_`` as ``str``, and
        ``probabilities`` the full probability row.

    Raises:
        ConfigurationError: if the pipeline lacks ``predict_proba``/``classes_``.
    """
    if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "classes_"):
        raise ConfigurationError(
            "this explainer requires a fitted classifier exposing predict_proba and "
            f"classes_; got {type(pipeline).__name__}"
        )
    probabilities = np.asarray(pipeline.predict_proba([text]))[0]
    index = int(np.argmax(probabilities))
    label = str(np.asarray(pipeline.classes_)[index])
    return index, label, probabilities
