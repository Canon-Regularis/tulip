"""The request and response protocol for the Claude dialect baseline.

Pure functions, free of the Anthropic SDK and the network, so they are unit tested
without either: building the system prompt and the message list a request carries,
resolving a model reply back to one known label, voting across self-consistency
variants, and content-addressing a request for the response cache.
:mod:`tulip.models.llm_baseline` composes these with the estimator and the SDK
client. Keeping the protocol here lets the prompt shape and the parsing rules
change without touching the classifier.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from tulip._serialize import sorted_json_text
from tulip.labels.taxonomy import display_name
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

logger = get_logger(__name__)

__all__ = ["build_messages", "build_system_prompt", "parse_label"]

#: Prompt-protocol version folded into every cache key. Bump it when the prompt or
#: message shape changes so stale cached responses are not reused.
_CACHE_VERSION = 1

_WORD_RE = re.compile(r"[a-z0-9]+")


def build_system_prompt(classes: Sequence[str]) -> str:
    """Build the instruction and label glossary the model classifies against.

    Args:
        classes: The label ids the model must choose between.

    Returns:
        A system prompt naming each label id with its English and Polish display
        names, so the model has the dialectology in front of it.
    """
    lines = [
        "You are a dialectologist classifying a short Polish text into exactly one "
        + "regional dialect.",
        "Choose the single best label id from this list. Reply with only the label id, "
        + "nothing else.",
        "",
        "Labels:",
    ]
    for label in classes:
        english = display_name(label)
        polish = display_name(label, polish=True)
        if polish and polish != english:
            lines.append(f"- {label}: {english} ({polish})")
        else:
            lines.append(f"- {label}: {english}")
    return "\n".join(lines)


def build_messages(text: str, exemplars: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """Build the message list: few-shot example turns, then the target text.

    Args:
        text: The document to classify.
        exemplars: ``(text, label)`` demonstrations, shown as prior turns.

    Returns:
        A messages list for the Messages API.
    """
    messages: list[dict[str, str]] = []
    for example_text, example_label in exemplars:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": example_label})
    messages.append({"role": "user", "content": str(text)})
    return messages


def parse_label(response_text: str, classes: Sequence[Any] | np.ndarray) -> str:
    """Resolve a model reply to one known label id, defensively.

    Tried in order: an exact (case-insensitive) match on the whole reply; a
    ``{"label": ...}`` JSON object (covering a structured-output reply); the label's
    words appearing as a contiguous phrase in the reply, preferring the most
    specific (longest) match so a compound id like ``cieszyn_silesia`` wins over an
    embedded sibling like ``silesia``. When none of those resolve, or the longest
    phrase match is ambiguous, the first class is returned so the classifier always
    yields a valid, deterministic label rather than raising.

    Args:
        response_text: The raw text the model returned.
        classes: The known label ids.

    Returns:
        One label id from ``classes``.
    """
    class_list = [str(label) for label in classes]
    text = response_text.strip()
    lowered = text.lower()

    for label in class_list:
        if lowered == label.lower():
            return label

    json_label = _json_label(text)
    if json_label is not None:
        for label in class_list:
            if json_label.lower() == label.lower():
                return label

    # Words, not underscore-tokens, so the display form ("Cieszyn Silesia") and the
    # id form ("cieszyn_silesia") tokenise the same way and both resolve.
    reply_words = _WORD_RE.findall(lowered)
    matches = [label for label in class_list if _contains_phrase(reply_words, _label_words(label))]
    if matches:
        longest = max(len(_label_words(label)) for label in matches)
        best = [label for label in matches if len(_label_words(label)) == longest]
        if len(best) == 1:
            return best[0]

    logger.debug("could not resolve LLM reply %r to a label; falling back", text[:80])
    return class_list[0]


def _label_words(label: str) -> tuple[str, ...]:
    """Split a label id into its lowercased word components."""
    return tuple(_WORD_RE.findall(label.lower()))


def _contains_phrase(words: Sequence[str], phrase: tuple[str, ...]) -> bool:
    """Whether ``phrase`` occurs as a contiguous run inside ``words``."""
    length = len(phrase)
    if length == 0:
        return False
    return any(tuple(words[i : i + length]) == phrase for i in range(len(words) - length + 1))


def _json_label(text: str) -> str | None:
    """Return the ``label`` field of a JSON-object reply, or ``None``."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict) and "label" in parsed:
        return str(parsed["label"])
    return None


def _majority_vote(votes: Sequence[str], classes: Sequence[Any] | np.ndarray) -> str:
    """The most-voted label, ties broken by class order for a deterministic result.

    ``votes`` are all valid label ids (:func:`parse_label` always returns one), so a
    simple count suffices; ``max`` keeps the first class in ``classes`` order on a
    tie.
    """
    counts = Counter(votes)
    class_list = [str(label) for label in classes]
    return max(class_list, key=lambda label: counts.get(label, 0))


def _cache_key(model: str, max_tokens: int, system: str, messages: Sequence[dict[str, str]]) -> str:
    """Content-address a request by everything that determines its response."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": list(messages),
        "v": _CACHE_VERSION,
    }
    return hashlib.sha256(sorted_json_text(payload).encode("utf-8")).hexdigest()
