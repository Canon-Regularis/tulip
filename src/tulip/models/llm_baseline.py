"""A constrained-choice LLM baseline for dialect classification.

:class:`LLMClassifier` asks Claude to label a short Polish text with exactly one
dialect from the taxonomy, wrapped in the scikit-learn estimator API so it drops
into ``benchmark`` and ``train`` as ``-m llm_zeroshot`` or ``-m llm_fewshot``
alongside the classical and neural models. Zero-shot sends only the label
glossary; few-shot prepends a few seeded worked examples per class.

The determinism problem this baseline poses is real: an API call is
network-bound and its output is not reproducible from a seed. Current Claude
models do not accept a sampling temperature, so there is no in-band knob to pin
the output either. The reproducibility boundary is therefore a content-addressed
response cache. Every request is keyed by a digest of the model id, the system
prompt, and the full message list; the first run records each response, and any
later run replays them, byte-identical and fully offline. A run that has to hit
the network (a cache miss) is non-deterministic and must never feed a committed
artifact; the leaderboard's byte-for-byte guarantee is preserved by keeping this
baseline out of the committed suite and gating it behind a populated cache.

``anthropic`` is an optional extra, imported lazily inside the one method that
calls the API, so importing this module (and registering the baseline) never
pulls the SDK. The prompt-building and response-parsing helpers are pure
functions, unit-testable without the SDK or a network.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tulip._serialize import sorted_json_text, write_sorted_json
from tulip.core.exceptions import ConfigurationError, TulipError
from tulip.labels.taxonomy import display_name
from tulip.models._common import (
    ArgmaxPredictMixin,
    empty_proba,
    reconcile_seed_param,
    require_fitted,
    validate_fit_inputs,
)
from tulip.models._llm_exemplars import EXEMPLAR_SELECTORS
from tulip.models.registry import MODELS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

#: Default Claude model: the balanced tier, a sensible cost/latency/quality point
#: for a baseline over many short texts. Override via the ``model`` parameter.
DEFAULT_MODEL = "claude-sonnet-5"

#: Output cap: a label id is a few tokens, so a small ceiling keeps the model
#: terse and the cost down.
DEFAULT_MAX_TOKENS = 64

#: The pip extra that provides the SDK, named in the install hint and the
#: registry metadata.
ANTHROPIC_EXTRA = "anthropic"

#: Prompt-protocol version folded into every cache key. Bump it when the prompt
#: or message shape changes so stale cached responses are not reused.
_CACHE_VERSION = 1

_WORD_RE = re.compile(r"[a-z0-9]+")

__all__ = [
    "ANTHROPIC_EXTRA",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "LLMClassifier",
    "LLMResponseCache",
    "build_messages",
    "build_system_prompt",
    "parse_label",
]


# --------------------------------------------------------------- pure helpers


def build_system_prompt(classes: Sequence[str]) -> str:
    """Build the instruction and label glossary the model classifies against.

    Args:
        classes: The label ids the model must choose between.

    Returns:
        A system prompt naming each label id with its English and Polish
        display names, so the model has the dialectology in front of it.
    """
    lines = [
        "You are a dialectologist classifying a short Polish text into exactly one "
        "regional dialect.",
        "Choose the single best label id from this list. Reply with only the label id, "
        "nothing else.",
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
    ``{"label": ...}`` JSON object (covering a structured-output reply); the
    label's words appearing as a contiguous phrase in the reply, preferring the
    most specific (longest) match so a compound id like ``cieszyn_silesia`` wins
    over an embedded sibling like ``silesia``. When none of those resolve, or the
    longest phrase match is ambiguous, the first class is returned so the
    classifier always yields a valid, deterministic label rather than raising.

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

    # Words, not underscore-tokens, so the display form ("Cieszyn Silesia") and
    # the id form ("cieszyn_silesia") tokenise the same way and both resolve.
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

    ``votes`` are all valid label ids (:func:`parse_label` always returns one),
    so a simple count suffices; ``max`` keeps the first class in ``classes`` order
    on a tie.
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


# --------------------------------------------------------------- response cache


class LLMResponseCache:
    """A content-addressed store of model responses, keyed by request digest.

    An in-memory layer backs an optional on-disk directory. With a directory,
    responses survive across processes, which is what lets a second run replay
    them offline; without one, the cache lives only for the process (useful in
    tests). Files are written with :func:`write_sorted_json`, so a committed
    cache stays diff-friendly and regenerable.
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        self._memory: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Return the cached response for ``key``, or ``None`` on a miss.

        A corrupt or partial cache file (invalid JSON, or missing the
        ``response`` key) is treated as a miss, so one bad entry re-queries and
        overwrites itself rather than aborting the whole prediction.
        """
        if key in self._memory:
            return self._memory[key]
        if self.directory is not None:
            path = self.directory / f"{key}.json"
            if path.is_file():
                try:
                    response = str(json.loads(path.read_text(encoding="utf-8"))["response"])
                except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
                    logger.debug("ignoring unreadable cache file %s: %s", path, exc)
                    return None
                self._memory[key] = response
                return response
        return None

    def put(self, key: str, response: str, *, request: dict[str, Any] | None = None) -> None:
        """Record ``response`` for ``key`` in memory and (if set) on disk."""
        self._memory[key] = response
        if self.directory is not None:
            payload: dict[str, Any] = {"response": response}
            if request is not None:
                payload["request"] = request
            write_sorted_json(self.directory / f"{key}.json", payload)


# --------------------------------------------------------------- classifier


class LLMClassifier(ArgmaxPredictMixin, ClassifierMixin, BaseEstimator):
    """A constrained-choice Claude classifier with a scikit-learn interface.

    ``fit`` records the label set and (for few-shot) a seeded set of worked
    examples; it makes no API calls. ``predict``/``predict_proba`` classify each
    text with one constrained-choice request, served from the response cache when
    possible. Probabilities are one-hot on the chosen label: this is a hard
    classifier, not a calibrated one.

    Attributes:
        classes_: Sorted array of class labels (after ``fit``).
        exemplars_: The few-shot demonstrations (after ``fit``).
        system_prompt_: The instruction and glossary sent as the system prompt.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        few_shot: int = 0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache_dir: Path | str | None = None,
        system_prompt: str | None = None,
        exemplar_selection: str = "random",
        self_consistency: int = 1,
        seed: int = 42,
    ) -> None:
        """Configure the wrapper; the SDK is imported only when the API is called.

        Args:
            model: Claude model id.
            few_shot: Worked examples per class to include (0 is zero-shot).
            max_tokens: Output token cap per request.
            cache_dir: Directory backing the response cache; ``None`` caches only
                in memory (so a fresh process re-queries).
            system_prompt: Override for the auto-built instruction and glossary.
            exemplar_selection: Few-shot selection strategy (``"random"`` or
                ``"similar"``; see :data:`~tulip.models._llm_exemplars.EXEMPLAR_SELECTORS`).
                ``"similar"`` picks per-class examples closest to the query text.
            self_consistency: Number of prompt variants to classify and
                majority-vote (1 disables it). Diversity comes from asking each
                variant for a different exemplar set, so it only helps with
                ``few_shot > 0``.
            seed: Seed for the deterministic few-shot example selection.

        Note:
            Per the scikit-learn estimator contract, ``__init__`` only stores
            parameters; validation happens in :meth:`fit`.
        """
        self.model = model
        self.few_shot = few_shot
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.system_prompt = system_prompt
        self.exemplar_selection = exemplar_selection
        self.self_consistency = self_consistency
        self.seed = seed

    def _validate_hyperparameters(self) -> None:
        """Validate constructor/set_params values (called from :meth:`fit`)."""
        if self.few_shot < 0:
            raise ConfigurationError(f"few_shot must be >= 0, got {self.few_shot}")
        if self.max_tokens < 1:
            raise ConfigurationError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.self_consistency < 1:
            raise ConfigurationError(f"self_consistency must be >= 1, got {self.self_consistency}")
        if self.exemplar_selection not in set(EXEMPLAR_SELECTORS.names()):
            options = ", ".join(EXEMPLAR_SELECTORS.names())
            raise ConfigurationError(
                f"unknown exemplar_selection {self.exemplar_selection!r}; choose from: {options}"
            )

    def fit(self, X: Sequence[str], y: Sequence[Any]) -> LLMClassifier:
        """Record the label set and few-shot examples; no API calls are made.

        Args:
            X: Sequence of raw documents.
            y: Parallel sequence of labels (coerced to str).

        Returns:
            ``self``, fitted.

        Raises:
            ConfigurationError: if a hyperparameter is out of range.
            DataError: if inputs are empty, mismatched, or single-class.
        """
        self._validate_hyperparameters()
        texts = [str(text) for text in X]
        classes, encoded = validate_fit_inputs(texts, y)
        labels = [str(label) for label in classes[encoded]]
        class_ids = [str(label) for label in classes]
        self.classes_ = classes
        self.system_prompt_ = self.system_prompt or build_system_prompt(class_ids)

        by_label: dict[str, list[str]] = {}
        for text, label in zip(texts, labels, strict=True):
            by_label.setdefault(label, []).append(text)
        self.selector_ = EXEMPLAR_SELECTORS.create(self.exemplar_selection).fit(
            by_label, class_ids, self.few_shot, self.seed
        )
        # A representative (variant 0) snapshot, so the query-independent random
        # selection stays inspectable and the value is defined for every strategy.
        self.exemplars_ = self.selector_.select("", variant=0)
        return self

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Return a one-hot class distribution for each text.

        Args:
            X: Sequence of raw documents.

        Returns:
            Array of shape ``(len(X), n_classes)``; columns follow ``classes_``.

        Raises:
            TulipError: if the model has not been fitted, or an API call fails.
        """
        require_fitted(self, "classes_")
        texts = [str(text) for text in X]
        n_classes = len(self.classes_)
        if not texts:
            return empty_proba(n_classes)
        class_to_index = {str(label): index for index, label in enumerate(self.classes_)}
        rows: list[np.ndarray] = []
        for text in texts:
            label = self._classify(text)
            row = np.zeros(n_classes, dtype=np.float64)
            row[class_to_index[label]] = 1.0
            rows.append(row)
        return np.vstack(rows)

    def _classify(self, text: str) -> str:
        """Classify one text; a single request, or a self-consistency vote.

        With ``self_consistency == 1`` this is one constrained-choice request
        (variant 0), identical to the base behaviour. Above 1, the same text is
        classified through several prompt variants (each a different exemplar
        set) and the labels are majority-voted, ties broken by class order.
        """
        if self.self_consistency <= 1:
            return self._classify_variant(text, variant=0)
        votes = [self._classify_variant(text, variant=v) for v in range(self.self_consistency)]
        return _majority_vote(votes, self.classes_)

    def _classify_variant(self, text: str, *, variant: int) -> str:
        """Classify one text with one prompt variant, from the cache when possible."""
        exemplars = self.selector_.select(text, variant=variant)
        messages = build_messages(text, exemplars)
        key = _cache_key(self.model, self.max_tokens, self.system_prompt_, messages)
        cached = self._cache.get(key)
        if cached is not None:
            return parse_label(cached, self.classes_)
        response = _complete(
            self._ensure_client(),
            model=self.model,
            system=self.system_prompt_,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        self._cache.put(
            key,
            response,
            request={"model": self.model, "system": self.system_prompt_, "messages": messages},
        )
        return parse_label(response, self.classes_)

    @property
    def _cache(self) -> LLMResponseCache:
        """The response cache, rebuilt lazily and after a ``cache_dir`` change."""
        desired = Path(self.cache_dir) if self.cache_dir is not None else None
        cache = getattr(self, "_cache_", None)
        if cache is None or cache.directory != desired:
            cache = self._cache_ = LLMResponseCache(self.cache_dir)
        return cache

    def _ensure_client(self) -> Any:
        """Build the Anthropic client lazily; never touched on a full cache hit."""
        client = getattr(self, "_client_", None)
        if client is None:
            anthropic = optional_import(
                "anthropic", extra=ANTHROPIC_EXTRA, purpose="LLM dialect classification baseline"
            )
            client = self._client_ = anthropic.Anthropic()
        return client

    def __getstate__(self) -> dict[str, Any]:
        # The live client and cache are runtime-only; drop them so a fitted
        # classifier pickles, and rebuild them lazily after load.
        state = dict(self.__dict__)
        state.pop("_client_", None)
        state.pop("_cache_", None)
        return state


def _complete(
    client: Any, *, model: str, system: str, messages: Sequence[dict[str, str]], max_tokens: int
) -> str:
    """Call the Messages API and return the reply text, wrapping SDK errors."""
    try:
        response = client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=list(messages)
        )
    except Exception as exc:
        # Network/SDK boundary: any SDK or transport error is re-raised as a
        # clean tulip error rather than surfacing a raw traceback.
        raise TulipError(f"Anthropic API call failed: {exc}") from exc
    return _extract_text(response)


def _extract_text(response: Any) -> str:
    """Join the text blocks of a Messages API response."""
    parts = [
        block.text
        for block in (getattr(response, "content", None) or [])
        if getattr(block, "text", None)
    ]
    return "".join(parts)


# raw_input: the baseline consumes raw texts directly, so it takes no extractors.
@MODELS.register("llm_zeroshot", metadata={"raw_input": True, "extra": ANTHROPIC_EXTRA})
def make_llm_zeroshot(**params: Any) -> LLMClassifier:
    """Create a zero-shot :class:`LLMClassifier` (``random_state`` aliases ``seed``)."""
    reconcile_seed_param(params)
    params.setdefault("few_shot", 0)
    return LLMClassifier(**params)


@MODELS.register("llm_fewshot", metadata={"raw_input": True, "extra": ANTHROPIC_EXTRA})
def make_llm_fewshot(**params: Any) -> LLMClassifier:
    """Create a few-shot :class:`LLMClassifier` (three examples per class by default)."""
    reconcile_seed_param(params)
    params.setdefault("few_shot", 3)
    return LLMClassifier(**params)
