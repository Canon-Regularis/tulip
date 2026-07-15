"""Tests for the constrained-choice LLM dialect baseline."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tulip.core.exceptions import MissingDependencyError, TulipError
from tulip.models import MODELS
from tulip.models.llm_baseline import (
    LLMClassifier,
    build_messages,
    build_system_prompt,
    parse_label,
)

_X = [
    "baca ma owce na hali",
    "ida bez pole we somy tukej",
    "juhas gra na hali baca",
    "we somy do dom bez pole",
]
_Y = ["podhale", "silesia", "podhale", "silesia"]


# --------------------------------------------------------------- fake client


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> _Response:
        self._client.calls += 1
        self._client.last_kwargs = kwargs
        text = str(kwargs["messages"][-1]["content"])
        label = "podhale" if "baca" in text or "hali" in text else "silesia"
        return _Response(label)


class _FakeClient:
    """A stand-in Anthropic client that records calls and never hits the network."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}
        self.messages = _Messages(self)


class _ExplodingClient:
    """A client whose create() raises, to exercise error wrapping."""

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    def __init__(self) -> None:
        self.messages = _ExplodingClient._Messages()


def _fitted(cache_dir: Path | None = None, *, few_shot: int = 0, seed: int = 0) -> LLMClassifier:
    clf = LLMClassifier(cache_dir=cache_dir, few_shot=few_shot, seed=seed)
    clf.fit(_X, _Y)
    return clf


# --------------------------------------------------------------- fit / predict


def test_fit_records_classes_and_makes_no_api_calls() -> None:
    clf = _fitted()
    clf._client_ = _ExplodingClient()  # any API call here would raise
    assert list(clf.classes_) == ["podhale", "silesia"]
    assert clf.exemplars_ == ()  # zero-shot has no demonstrations
    assert "podhale" in clf.system_prompt_


def test_zero_shot_predicts_and_proba_is_one_hot() -> None:
    clf = _fitted()
    clf._client_ = _FakeClient()
    predictions = clf.predict(_X)
    assert list(predictions) == _Y
    proba = clf.predict_proba(_X)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(proba)) <= {0.0, 1.0}


def test_no_sampling_temperature_is_sent() -> None:
    clf = _fitted()
    client = _FakeClient()
    clf._client_ = client
    clf.predict(_X[:1])
    assert "temperature" not in client.last_kwargs
    assert client.last_kwargs["model"] == "claude-sonnet-5"


def test_empty_input_returns_well_shaped_empty() -> None:
    clf = _fitted()
    clf._client_ = _ExplodingClient()
    proba = clf.predict_proba([])
    assert proba.shape == (0, 2)


# --------------------------------------------------------------- cache


def test_cache_makes_a_second_run_offline(tmp_path: Path) -> None:
    first = _fitted(tmp_path)
    client = _FakeClient()
    first._client_ = client
    predictions = first.predict(_X)
    assert client.calls == 4  # one request per text on the cold cache

    # A fresh classifier over the same cache directory, whose client would raise
    # if touched, reproduces the predictions without a single API call.
    second = _fitted(tmp_path)
    second._client_ = _ExplodingClient()
    assert list(second.predict(_X)) == list(predictions)


def test_cache_files_are_byte_stable(tmp_path: Path) -> None:
    a = _fitted(tmp_path / "a")
    a._client_ = _FakeClient()
    a.predict(_X)
    b = _fitted(tmp_path / "b")
    b._client_ = _FakeClient()
    b.predict(_X)

    files_a = sorted(p.name for p in (tmp_path / "a").glob("*.json"))
    files_b = sorted(p.name for p in (tmp_path / "b").glob("*.json"))
    assert files_a == files_b  # same content-addressed keys
    for name in files_a:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_corrupt_cache_file_is_treated_as_a_miss(tmp_path: Path) -> None:
    warm = _fitted(tmp_path)
    warm._client_ = _FakeClient()
    warm.predict(_X[:1])
    cache_file = next(tmp_path.glob("*.json"))
    cache_file.write_text("{ this is not valid json", encoding="utf-8")

    # A fresh classifier over the corrupted cache re-queries rather than crashing.
    recover = _fitted(tmp_path)
    client = _FakeClient()
    recover._client_ = client
    assert list(recover.predict(_X[:1])) == [_Y[0]]
    assert client.calls == 1  # the corrupt entry forced a re-query


def test_set_params_cache_dir_is_honoured_after_a_predict(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    clf = _fitted(dir_a)
    clf._client_ = _FakeClient()
    clf.predict(_X[:1])  # binds the cache to dir_a

    clf.set_params(cache_dir=dir_b)
    clf.predict(_X[:1])  # must now write into dir_b
    assert dir_b.is_dir()
    assert list(dir_b.glob("*.json"))


def test_a_full_cache_hit_never_builds_a_client(tmp_path: Path) -> None:
    warm = _fitted(tmp_path)
    warm._client_ = _FakeClient()
    warm.predict(_X)

    # _ensure_client would raise; a fully cached predict must not reach it.
    cold = _fitted(tmp_path)

    def _boom() -> Any:
        raise AssertionError("client must not be built on a full cache hit")

    cold._ensure_client = _boom  # type: ignore[method-assign]
    assert list(cold.predict(_X)) == _Y


# --------------------------------------------------------------- few-shot


def test_few_shot_selection_is_seeded_and_deterministic(tmp_path: Path) -> None:
    a = _fitted(few_shot=1, seed=7)
    b = _fitted(few_shot=1, seed=7)
    assert a.exemplars_ == b.exemplars_
    assert len(a.exemplars_) == 2  # one example per class
    labels = {label for _, label in a.exemplars_}
    assert labels == {"podhale", "silesia"}


def test_few_shot_examples_reach_the_messages() -> None:
    clf = _fitted(few_shot=1, seed=7)
    messages = build_messages("nowy tekst", clf.exemplars_)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[-1] == {"role": "user", "content": "nowy tekst"}


# --------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("podhale", "podhale"),
        ("  Podhale\n", "podhale"),
        ('{"label": "silesia"}', "silesia"),
        ("I think this is podhale.", "podhale"),
        ("could be podhale or silesia", "podhale"),  # ambiguous -> first class
        ("no idea at all", "podhale"),  # unresolved -> first class
    ],
)
def test_parse_label(reply: str, expected: str) -> None:
    assert parse_label(reply, ["podhale", "silesia"]) == expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Cieszyn Silesia", "cieszyn_silesia"),  # display form of the compound id
        ("cieszyn silesia", "cieszyn_silesia"),
        ('{"label": "Cieszyn Silesia"}', "cieszyn_silesia"),
        ("the answer is cieszyn_silesia", "cieszyn_silesia"),  # embedded id form
        ("this is silesia", "silesia"),  # the bare sibling still resolves
    ],
)
def test_parse_label_prefers_the_compound_over_an_embedded_sibling(
    reply: str, expected: str
) -> None:
    # cieszyn_silesia embeds the sibling id silesia; the longest phrase must win.
    assert parse_label(reply, ["cieszyn_silesia", "silesia"]) == expected


def test_system_prompt_lists_labels_with_a_gloss() -> None:
    prompt = build_system_prompt(["podhale", "silesia"])
    assert "- podhale:" in prompt
    assert "- silesia:" in prompt
    assert "exactly one" in prompt


# --------------------------------------------------------------- errors / persistence


def test_api_error_is_wrapped_as_a_tulip_error() -> None:
    clf = _fitted()
    clf._client_ = _ExplodingClient()
    with pytest.raises(TulipError, match="Anthropic API call failed"):
        clf.predict(_X)


def test_cache_miss_without_the_sdk_raises_missing_dependency(monkeypatch: Any) -> None:
    clf = _fitted()

    def _no_sdk(*args: Any, **kwargs: Any) -> Any:
        raise MissingDependencyError("anthropic", extra="anthropic")

    monkeypatch.setattr("tulip.models.llm_baseline.optional_import", _no_sdk)
    with pytest.raises(MissingDependencyError):
        clf.predict(_X)


def test_pickle_drops_the_client_and_stays_offline(tmp_path: Path) -> None:
    clf = _fitted(tmp_path)
    clf._client_ = _FakeClient()
    predictions = clf.predict(_X)  # warm the on-disk cache

    restored = pickle.loads(pickle.dumps(clf))
    assert "_client_" not in restored.__dict__
    assert list(restored.classes_) == ["podhale", "silesia"]
    restored._client_ = _ExplodingClient()  # would raise if the API were hit
    assert list(restored.predict(_X)) == list(predictions)


def test_invalid_hyperparameters_are_rejected() -> None:
    with pytest.raises(Exception, match="few_shot"):
        LLMClassifier(few_shot=-1).fit(_X, _Y)
    with pytest.raises(Exception, match="max_tokens"):
        LLMClassifier(max_tokens=0).fit(_X, _Y)


# --------------------------------------------------------------- registry / policy


def test_registered_as_raw_input_optional_models() -> None:
    for name in ("llm_zeroshot", "llm_fewshot"):
        metadata = MODELS.metadata(name)
        assert metadata["raw_input"] is True
        assert metadata["extra"] == "anthropic"


def test_few_shot_factory_default() -> None:
    zero = MODELS.create("llm_zeroshot")
    few = MODELS.create("llm_fewshot")
    assert zero.few_shot == 0
    assert few.few_shot == 3


def test_baseline_is_absent_from_committed_configs() -> None:
    # The LLM baseline is non-deterministic on a cache miss, so it must never
    # sit in a committed suite that feeds the byte-stable leaderboard.
    roots = [Path("benchmarks"), Path("configs")]
    configs = [path for root in roots if root.is_dir() for path in root.rglob("*.yaml")]
    assert configs, "expected committed config files to scan"
    offenders = [path for path in configs if "llm_" in path.read_text(encoding="utf-8")]
    assert not offenders, f"LLM baseline leaked into committed configs: {offenders}"


def test_import_pulls_no_sdk() -> None:
    code = (
        "import sys, tulip.models.llm_baseline, tulip.data;"
        "raise SystemExit(1 if 'anthropic' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)  # noqa: S603  (trusted, fixed input)
    assert result.returncode == 0
