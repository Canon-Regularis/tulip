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


def predicted_class(pipeline: Any, text: str) -> tuple[int, str, list[str]]:
    """Return the predicted class for ``text`` under a probabilistic pipeline.

    Args:
        pipeline: A fitted object exposing ``predict_proba`` and ``classes_``.
        text: The raw input document.

    Returns:
        ``(index, label, class_names)`` where ``index`` is the argmax column,
        ``label`` the corresponding entry of ``classes_`` as ``str``, and
        ``class_names`` every entry of ``classes_`` as ``str``.

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
    class_names = [str(label) for label in np.asarray(pipeline.classes_)]
    return index, class_names[index], class_names


def predicted_label_or_none(pipeline: Any, text: str) -> str | None:
    """Return ``pipeline``'s predicted label for ``text``, or ``None``.

    Returns ``None`` when ``pipeline`` is ``None`` or does not expose
    ``predict``. The prediction call itself is left unguarded; callers that
    need to tolerate a failing ``predict`` should wrap this in their own
    try/except.

    Args:
        pipeline: An optional fitted classifier exposing ``predict``.
        text: The raw input document.

    Returns:
        The predicted label as ``str``, or ``None`` if no prediction is
        available.
    """
    if pipeline is None or not hasattr(pipeline, "predict"):
        return None
    return str(pipeline.predict([text])[0])
