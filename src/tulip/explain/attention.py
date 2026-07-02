"""Attention-based explanations for transformer text classifiers.

Runs one forward pass with ``output_attentions=True`` and reports, for each
input word, the attention mass flowing from the classification anchor token
([CLS] or the first token) to that word in the chosen layer, averaged over
heads. Subword pieces are merged into whole words (weights summed) so the
output is readable to a dialectologist rather than a tokenizer.

Caveat, stated up front: attention weights show what the model *looked at*,
not a causal attribution of the decision — treat them as a complementary
signal next to ``lime``/``shap``, not a replacement.

Works with any object exposing fitted ``model_`` (a Hugging Face sequence
classifier) and ``tokenizer_`` attributes — in tulip that is
:class:`tulip.models.neural_text.TransformerTextClassifier`. torch is an
optional dependency imported lazily inside ``explain``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import Explanation, TokenAttribution
from tulip.explain._shared import as_text
from tulip.explain.registry import EXPLAINERS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

logger = get_logger(__name__)

__all__ = ["AttentionExplainer"]

#: Subword markers, by scheme: WordPiece continuation prefix, SentencePiece
#: and byte-level BPE word-start prefixes, and the HerBERT-style end-of-word
#: suffix.
_WORDPIECE_PREFIX = "##"
_START_MARKERS = ("▁", "Ġ")  # "▁" (SentencePiece), "Ġ" (byte-level BPE)
_END_OF_WORD_SUFFIX = "</w>"


def _clean_piece(piece: str) -> str:
    """Strip subword markers from one token piece for display."""
    if piece.startswith(_WORDPIECE_PREFIX):
        piece = piece[len(_WORDPIECE_PREFIX) :]
    for marker in _START_MARKERS:
        if piece.startswith(marker):
            piece = piece[len(marker) :]
    if piece.endswith(_END_OF_WORD_SUFFIX):
        piece = piece[: -len(_END_OF_WORD_SUFFIX)]
    return piece


def _merge_by_word_ids(
    text: str, encoding: Any, weights: np.ndarray
) -> list[tuple[str, float]] | None:
    """Merge subword weights into words using a fast tokenizer's alignment.

    Fast tokenizers expose ``word_ids`` and character offsets, giving an
    exact, scheme-independent mapping from token pieces to source words; the
    word text is sliced straight from the original input so no marker
    stripping heuristics are needed.

    Returns:
        ``(word, summed weight)`` pairs, or ``None`` when the tokenizer does
        not provide word alignment (slow tokenizers).
    """
    try:
        word_ids = encoding.word_ids(0)
    except (AttributeError, ValueError):
        return None
    if word_ids is None or all(word_id is None for word_id in word_ids):
        return None
    offsets = encoding.get("offset_mapping")
    if offsets is None:
        return None
    spans = np.asarray(offsets)[0]

    merged: list[tuple[str, float]] = []
    current_id: int | None = None
    start = end = 0
    weight_sum = 0.0
    for position, word_id in enumerate(word_ids):
        if word_id is None:  # special token: closes any open word
            if current_id is not None:
                merged.append((text[start:end], weight_sum))
                current_id = None
            continue
        if word_id != current_id:
            if current_id is not None:
                merged.append((text[start:end], weight_sum))
            current_id = word_id
            start = int(spans[position][0])
            end = int(spans[position][1])
            weight_sum = 0.0
        else:
            end = int(spans[position][1])
        weight_sum += float(weights[position])
    if current_id is not None:
        merged.append((text[start:end], weight_sum))
    return merged


def _merge_by_markers(
    tokens: list[str], weights: np.ndarray, special_tokens: set[str]
) -> list[tuple[str, float]]:
    """Merge subword weights into words from token markers (fallback path).

    Infers the subword scheme from the pieces themselves: explicit word-start
    prefixes ("▁"/"Ġ") mean unmarked pieces continue the previous word;
    otherwise WordPiece rules apply ("##" continues, anything else starts a
    new word), with "</w>" suffixes forcing a word boundary. Imperfect for
    exotic tokenizers, but only used when fast-tokenizer alignment is
    unavailable.
    """
    prefix_scheme = any(token.startswith(_START_MARKERS) for token in tokens)
    merged: list[tuple[str, float]] = []
    open_word = False
    for token, weight in zip(tokens, weights, strict=True):
        if token in special_tokens:
            open_word = False
            continue
        if prefix_scheme:
            starts_new = token.startswith(_START_MARKERS)
        else:
            starts_new = not token.startswith(_WORDPIECE_PREFIX)
        piece = _clean_piece(token)
        if open_word and not starts_new and merged:
            previous_piece, previous_weight = merged[-1]
            merged[-1] = (previous_piece + piece, previous_weight + float(weight))
        else:
            merged.append((piece, float(weight)))
        open_word = not token.endswith(_END_OF_WORD_SUFFIX)
    return merged


@EXPLAINERS.register("attention")
class AttentionExplainer:
    """Per-word attention mass from the classification anchor token.

    Attributes:
        layer: Index of the attention layer to attribute from (default ``-1``,
            the last layer, whose [CLS] representation feeds the classifier
            head).
    """

    def __init__(self, layer: int = -1) -> None:
        """Configure the explainer.

        Args:
            layer: Attention layer to read (negative indices count from the
                end, as in Python indexing).
        """
        self.layer = layer

    def explain(self, pipeline: Any, raw_input: Any, **kwargs: Any) -> Explanation:
        """Explain one prediction from the model's attention weights.

        Args:
            pipeline: A fitted transformer classifier exposing ``model_`` (a
                Hugging Face sequence-classification model) and ``tokenizer_``
                — e.g. :class:`tulip.models.neural_text.TransformerTextClassifier`.
            raw_input: The raw document to explain.
            **kwargs: ``layer`` overrides the constructor value.

        Returns:
            An :class:`Explanation` with one :class:`TokenAttribution` per
            input word (attention mass from [CLS], averaged over heads, summed
            over subword pieces) and a per-layer summary in ``details``:
            ``details["layers"]`` holds, per layer, the head-averaged
            CLS-to-token attention over the raw subword tokens
            (``details["tokens"]``).

        Raises:
            ConfigurationError: if ``pipeline`` does not expose a fitted
                ``model_``/``tokenizer_`` pair.
            MissingDependencyError: if torch is not installed.
        """
        model = getattr(pipeline, "model_", None)
        tokenizer = getattr(pipeline, "tokenizer_", None)
        if model is None or tokenizer is None:
            raise ConfigurationError(
                "the attention explainer requires a fitted transformer classifier exposing "
                f"model_ and tokenizer_ (e.g. TransformerTextClassifier); got "
                f"{type(pipeline).__name__}. For sklearn pipelines use 'top_tfidf', 'lime', "
                "or 'shap' instead."
            )
        torch = optional_import(
            "torch", extra="transformers", purpose="attention-based explanations"
        )
        layer = int(kwargs.get("layer", self.layer))
        text = as_text(raw_input)
        device = getattr(pipeline, "device_", None) or "cpu"
        max_length = int(getattr(pipeline, "max_length", 512))

        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=bool(getattr(tokenizer, "is_fast", False)),
        )
        inputs = {
            key: value.to(device)
            for key, value in encoding.items()
            if key != "offset_mapping"  # alignment metadata, not a model input
        }
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions  # per layer: (1, heads, seq, seq)
        n_layers = len(attentions)
        if not -n_layers <= layer < n_layers:
            raise ConfigurationError(
                f"layer {layer} is out of range for a model with {n_layers} attention layers"
            )
        # Attention mass from the anchor (position 0: [CLS] or first token) to
        # every position, averaged over heads.
        per_layer = [
            attention[0, :, 0, :].mean(dim=0).detach().cpu().numpy().astype(np.float64)
            for attention in attentions
        ]
        weights = per_layer[layer]

        token_ids = encoding["input_ids"][0].tolist()
        tokens = [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]

        merged = _merge_by_word_ids(text, encoding, weights)
        if merged is None:
            special_tokens = {str(token) for token in tokenizer.all_special_tokens}
            merged = _merge_by_markers(tokens, weights, special_tokens)
        attributions = tuple(
            TokenAttribution(token=word, weight=weight) for word, weight in merged if word
        )

        predicted_label = self._predicted_label(pipeline, model, outputs)
        return Explanation(
            method="attention",
            predicted_label=predicted_label,
            attributions=attributions,
            details={
                "tokens": tokens,
                "layers": [layer_weights.tolist() for layer_weights in per_layer],
                "layer_used": layer,
                "num_layers": n_layers,
                "num_heads": int(attentions[0].shape[1]),
            },
        )

    @staticmethod
    def _predicted_label(pipeline: Any, model: Any, outputs: Any) -> str | None:
        """Resolve the predicted label from logits via classes_ or id2label."""
        logits = outputs.logits.detach().cpu().numpy()
        index = int(np.argmax(logits[0]))
        classes = getattr(pipeline, "classes_", None)
        if classes is not None and index < len(classes):
            return str(np.asarray(classes)[index])
        id2label = getattr(getattr(model, "config", None), "id2label", None)
        if id2label:
            return str(id2label.get(index, index))
        return None
