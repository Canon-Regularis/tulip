"""Tests for the LLM baseline upgrades: exemplar selection and self-consistency."""

from __future__ import annotations

from typing import Any

import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.models._llm_exemplars import (
    EXEMPLAR_SELECTORS,
    RandomExemplarSelector,
    SimilarExemplarSelector,
)
from tulip.models.llm_baseline import LLMClassifier, _majority_vote

_X = [
    "baca ma owce na hali",
    "ida bez pole we somy tukej",
    "juhas gra na hali baca",
    "we somy do dom bez pole",
]
_Y = ["podhale", "silesia", "podhale", "silesia"]


# --------------------------------------------------------------- selectors


def test_selectors_are_registered() -> None:
    assert set(EXEMPLAR_SELECTORS.names()) == {"random", "similar"}


def test_random_selector_variant_0_is_a_stable_seeded_draw() -> None:
    pools = {"a": ["a1", "a2", "a3", "a4"], "b": ["b1", "b2", "b3", "b4"]}
    first = RandomExemplarSelector().fit(pools, ["a", "b"], few_shot=1, seed=7)
    second = RandomExemplarSelector().fit(pools, ["a", "b"], few_shot=1, seed=7)
    assert first.select("q", variant=0) == second.select("q", variant=0)
    assert {label for _, label in first.select("q", variant=0)} == {"a", "b"}


def test_random_selector_variants_draw_differently() -> None:
    pools = {"a": ["a1", "a2", "a3", "a4"], "b": ["b1", "b2", "b3", "b4"]}
    selector = RandomExemplarSelector().fit(pools, ["a", "b"], few_shot=1, seed=7)
    variants = {selector.select("q", variant=v) for v in range(4)}
    assert len(variants) >= 2  # self-consistency gets prompt diversity


def test_random_selector_zero_shot_is_empty() -> None:
    selector = RandomExemplarSelector().fit({"a": ["a1"]}, ["a"], few_shot=0, seed=0)
    assert selector.select("q") == ()


def test_similar_selector_picks_the_closest_example_per_class() -> None:
    pools = {
        "podhale": ["baca owce na hali", "juhas gra wesolo"],
        "silesia": ["gruba i wongiel", "sztajger na szychcie"],
    }
    selector = SimilarExemplarSelector().fit(pools, ["podhale", "silesia"], few_shot=1, seed=0)
    picked = selector.select("baca na hali dzis", variant=0)
    assert len(picked) == 2  # one per class
    # The query shares "baca"/"hali" with the first podhale example.
    assert ("baca owce na hali", "podhale") in picked


def test_similar_selector_variant_slides_the_window() -> None:
    pools = {"a": ["a matches query", "a second", "a third"]}
    selector = SimilarExemplarSelector().fit(pools, ["a"], few_shot=1, seed=0)
    top = selector.select("query", variant=0)
    nxt = selector.select("query", variant=1)
    assert top != nxt  # a later variant draws a different band of the ranking


# --------------------------------------------------------------- majority vote


@pytest.mark.parametrize(
    ("votes", "classes", "expected"),
    [
        (["a", "b", "a"], ["a", "b"], "a"),  # clear majority
        (["b", "b", "a"], ["a", "b"], "b"),
        (["a", "b"], ["a", "b"], "a"),  # tie -> first class
        (["b", "a"], ["a", "b"], "a"),  # tie -> first class regardless of vote order
    ],
)
def test_majority_vote(votes: list[str], classes: list[str], expected: str) -> None:
    assert _majority_vote(votes, classes) == expected


# --------------------------------------------------------------- classifier wiring


def test_self_consistency_votes_across_variants(monkeypatch: Any) -> None:
    clf = LLMClassifier(few_shot=1, seed=7, self_consistency=3).fit(_X, _Y)
    votes = ["silesia", "podhale", "podhale"]
    monkeypatch.setattr(clf, "_classify_variant", lambda text, *, variant: votes[variant])
    assert clf._classify("anything") == "podhale"  # 2 vs 1


def test_self_consistency_one_is_a_single_variant(monkeypatch: Any) -> None:
    clf = LLMClassifier(few_shot=1, seed=7, self_consistency=1).fit(_X, _Y)
    seen: list[int] = []

    def record(text: str, *, variant: int) -> str:
        seen.append(variant)
        return "podhale"

    monkeypatch.setattr(clf, "_classify_variant", record)
    clf._classify("x")
    assert seen == [0]  # exactly one variant, the base behaviour


def test_similar_selection_predicts(tmp_path) -> None:
    # End to end with the fake client, just to prove the wiring holds.
    from test_models_llm_baseline import _FakeClient

    clf = LLMClassifier(few_shot=1, seed=7, exemplar_selection="similar", cache_dir=tmp_path)
    clf.fit(_X, _Y)
    clf._client_ = _FakeClient()
    predictions = clf.predict(_X)
    assert set(predictions) <= {"podhale", "silesia"}


def test_invalid_upgrade_params_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="self_consistency"):
        LLMClassifier(self_consistency=0).fit(_X, _Y)
    with pytest.raises(ConfigurationError, match="exemplar_selection"):
        LLMClassifier(exemplar_selection="does_not_exist").fit(_X, _Y)
