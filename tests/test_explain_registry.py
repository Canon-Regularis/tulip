"""Registry-level tests for tulip.explain."""

from __future__ import annotations

import pytest

from tulip.core.exceptions import UnknownComponentError
from tulip.explain import EXPLAINERS, get_explainer

CANONICAL_NAMES = {
    "top_tfidf",
    "lime",
    "shap",
    "attention",
    "nearest_examples",
    "dialect_evidence",
}


def test_all_canonical_explainers_registered() -> None:
    assert set(EXPLAINERS.names()) == CANONICAL_NAMES


def test_get_explainer_returns_instances_with_explain() -> None:
    for name in CANONICAL_NAMES:
        explainer = get_explainer(name)
        assert callable(explainer.explain), name


def test_instantiation_never_requires_optional_dependencies() -> None:
    # Heavy deps (lime, shap, torch) must only be imported inside explain();
    # constructing every registered explainer must always succeed.
    for name in CANONICAL_NAMES:
        get_explainer(name)


def test_unknown_explainer_raises() -> None:
    with pytest.raises(UnknownComponentError):
        get_explainer("gradient_of_doom")


def test_constructor_kwargs_are_forwarded() -> None:
    explainer = get_explainer("top_tfidf", top_k=3)
    assert explainer.top_k == 3
    neighbors = get_explainer("nearest_examples", k=7)
    assert neighbors.k == 7
