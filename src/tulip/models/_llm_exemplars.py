"""Few-shot exemplar selection strategies for the LLM baseline.

The constrained-choice baseline (:mod:`tulip.models.llm_baseline`) shows the
model a few worked examples per class. *Which* examples matters: a fixed random
draw is cheap but blind, whereas picking examples similar to the query text puts
the most relevant demonstrations in front of the model. This module makes the
choice a pluggable strategy, mirroring the acquisition-strategy registry: a new
selector is a class plus a decorator, nothing central to edit.

Every selector also takes a ``variant`` index. Self-consistency runs the same
query through several prompt variants and votes; because the current Claude
models expose no sampling temperature, the diversity has to come from the
prompt, so each variant asks for a different exemplar set (a different random
draw, or a different band of the similarity ranking). Variant 0 is the plain
single-shot selection, so a self-consistency of 1 reproduces the base behaviour
byte for byte.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from tulip.core.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "EXEMPLAR_SELECTORS",
    "ExemplarSelector",
    "RandomExemplarSelector",
    "SimilarExemplarSelector",
]


@runtime_checkable
class ExemplarSelector(Protocol):
    """Chooses which worked examples to show for a query."""

    name: str

    def fit(
        self, by_label: Mapping[str, list[str]], class_ids: Sequence[str], few_shot: int, seed: int
    ) -> ExemplarSelector:
        """Record the per-class example pools and selection parameters."""

    def select(self, query: str, *, variant: int = 0) -> tuple[tuple[str, str], ...]:
        """Return ``(text, label)`` demonstrations for ``query`` and this variant."""


#: Canonical name -> exemplar selector class. ``EXEMPLAR_SELECTORS.create(name)``
#: returns a ready selector; call ``.fit(...)`` before ``.select(...)``.
EXEMPLAR_SELECTORS: Registry[type[ExemplarSelector]] = Registry("LLM exemplar selector")


@EXEMPLAR_SELECTORS.register("random")
class RandomExemplarSelector:
    """A seeded random draw of examples per class, independent of the query.

    Variant 0 reproduces the baseline's original selection exactly (one seeded
    generator consumed across the classes in id order, a sorted prefix per
    class); a later variant re-seeds to a different draw for self-consistency.
    """

    name = "random"

    def fit(
        self, by_label: Mapping[str, list[str]], class_ids: Sequence[str], few_shot: int, seed: int
    ) -> RandomExemplarSelector:
        self._by_label = dict(by_label)
        self._class_ids = tuple(class_ids)
        self._few_shot = few_shot
        self._seed = seed
        return self

    def select(self, query: str, *, variant: int = 0) -> tuple[tuple[str, str], ...]:
        del query  # a random draw ignores the query text
        if self._few_shot <= 0:
            return ()
        rng = np.random.default_rng(self._seed + variant)
        chosen: list[tuple[str, str]] = []
        for label in self._class_ids:
            pool = self._by_label.get(label, [])
            if not pool:
                continue
            k = min(self._few_shot, len(pool))
            picks = rng.permutation(len(pool))[:k]
            chosen.extend((pool[index], label) for index in sorted(picks))
        return tuple(chosen)


@EXEMPLAR_SELECTORS.register("similar")
class SimilarExemplarSelector:
    """Per class, the examples most similar to the query by character n-grams.

    Similarity is char-n-gram TF-IDF cosine (no embeddings, no network), fit on
    the labeled pool. Each class contributes its ``few_shot`` closest examples,
    so the model sees relevant demonstrations of every option. A self-consistency
    variant slides down the similarity ranking (the next band of close examples),
    keeping diversity without a temperature knob.
    """

    name = "similar"

    def fit(
        self, by_label: Mapping[str, list[str]], class_ids: Sequence[str], few_shot: int, seed: int
    ) -> SimilarExemplarSelector:
        del seed  # the ranking is deterministic; no randomness to seed
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._by_label = dict(by_label)
        self._class_ids = tuple(class_ids)
        self._few_shot = few_shot
        corpus = [text for texts in by_label.values() for text in texts] or [""]
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(corpus)
        self._vectors = {
            label: self._vectorizer.transform(texts)
            for label, texts in self._by_label.items()
            if texts
        }
        return self

    def select(self, query: str, *, variant: int = 0) -> tuple[tuple[str, str], ...]:
        if self._few_shot <= 0:
            return ()
        from sklearn.metrics.pairwise import cosine_similarity

        query_vector = self._vectorizer.transform([str(query)])
        chosen: list[tuple[str, str]] = []
        for label in self._class_ids:
            texts = self._by_label.get(label, [])
            if not texts:
                continue
            similarity = cosine_similarity(query_vector, self._vectors[label])[0]
            ranked = list(np.argsort(-similarity, kind="stable"))
            k = min(self._few_shot, len(texts))
            # A variant slides the window down the ranking (wrapping), so
            # self-consistency variants draw different but still-similar bands.
            start = (variant * k) % len(ranked)
            picks = [ranked[(start + offset) % len(ranked)] for offset in range(k)]
            chosen.extend((texts[index], label) for index in picks)
        return tuple(chosen)
