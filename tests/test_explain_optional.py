"""Tests for the lime, shap, and attention explainers (optional deps).

Each test skips cleanly when its optional dependency is not installed;
the attention duck-typing error path needs no optional dependency at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tulip.core.exceptions import ConfigurationError
from tulip.explain import get_explainer

PODHALE_QUERY = "Hej, baca się pyto, kaj się owce pasą na holi."


@pytest.fixture
def fitted_pipeline(synthetic_texts_and_labels: tuple[list[str], list[str]]) -> Pipeline:
    texts, labels = synthetic_texts_and_labels
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=2000, random_state=0)),
        ]
    )
    return pipeline.fit(texts, labels)


def _input_words(text: str) -> set[str]:
    import re

    return {word.lower() for word in re.findall(r"\w+", text, flags=re.UNICODE)}


# ---------------------------------------------------------------------------
# lime
# ---------------------------------------------------------------------------


def test_lime_attributions_come_from_the_input(fitted_pipeline: Pipeline) -> None:
    pytest.importorskip("lime")
    explanation = get_explainer("lime", num_samples=200, seed=0).explain(
        fitted_pipeline, PODHALE_QUERY
    )
    assert explanation.method == "lime"
    assert explanation.attributions
    words = _input_words(PODHALE_QUERY)
    for attribution in explanation.attributions:
        assert attribution.token.lower() in words
    assert explanation.predicted_label in explanation.details["class_names"]


def test_lime_is_deterministic_under_a_seed(fitted_pipeline: Pipeline) -> None:
    pytest.importorskip("lime")
    explainer = get_explainer("lime", num_samples=200, seed=7)
    first = explainer.explain(fitted_pipeline, PODHALE_QUERY)
    second = explainer.explain(fitted_pipeline, PODHALE_QUERY)
    assert [(a.token, a.weight) for a in first.attributions] == [
        (a.token, a.weight) for a in second.attributions
    ]


def test_lime_num_features_cap(fitted_pipeline: Pipeline) -> None:
    pytest.importorskip("lime")
    explanation = get_explainer("lime", num_samples=200, seed=0).explain(
        fitted_pipeline, PODHALE_QUERY, num_features=3
    )
    assert len(explanation.attributions) <= 3


# ---------------------------------------------------------------------------
# shap
# ---------------------------------------------------------------------------


def test_shap_attributions_come_from_the_input(fitted_pipeline: Pipeline) -> None:
    pytest.importorskip("shap")
    explanation = get_explainer("shap", max_evals=100).explain(fitted_pipeline, PODHALE_QUERY)
    assert explanation.method == "shap"
    assert explanation.attributions
    words = _input_words(PODHALE_QUERY)
    for attribution in explanation.attributions:
        assert attribution.token.lower() in words
    assert explanation.predicted_label in explanation.details["class_names"]


def test_shap_top_k_caps_attributions(fitted_pipeline: Pipeline) -> None:
    pytest.importorskip("shap")
    explanation = get_explainer("shap", max_evals=100, top_k=4).explain(
        fitted_pipeline, PODHALE_QUERY
    )
    assert 0 < len(explanation.attributions) <= 4


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def test_attention_rejects_non_transformer_pipelines(fitted_pipeline: Pipeline) -> None:
    # Duck-typing check fires before any optional import: no torch required.
    with pytest.raises(ConfigurationError, match="model_"):
        get_explainer("attention").explain(fitted_pipeline, PODHALE_QUERY)


def _tiny_bert(tmp_path):  # -> duck-typed TransformerTextClassifier stand-in
    torch = pytest.importorskip("torch")  # noqa: F841
    transformers = pytest.importorskip("transformers")

    pieces = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "kaj",
        "sie",
        "ow",
        "##ce",
        "pasa",
        "na",
        "holi",
        "baca",
    ]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(pieces), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    config = transformers.BertConfig(
        vocab_size=len(pieces),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=2,
    )
    model = transformers.BertForSequenceClassification(config)

    class Duck:
        model_ = model
        tokenizer_ = tokenizer
        classes_ = np.array(["podhale", "standard"])
        device_ = "cpu"
        max_length = 32

    return Duck()


def test_attention_merges_subwords_into_words(tmp_path) -> None:
    duck = _tiny_bert(tmp_path)
    explanation = get_explainer("attention").explain(duck, "kaj sie owce pasa na holi")
    assert explanation.method == "attention"
    tokens = [attribution.token for attribution in explanation.attributions]
    assert "owce" in tokens  # "ow" + "##ce" merged back into one word
    assert not any(token.startswith("##") for token in tokens)
    assert "[CLS]" not in tokens
    assert "[SEP]" not in tokens
    assert explanation.predicted_label in {"podhale", "standard"}
    # Attention rows are softmax-normalised: word masses (plus the mass on
    # special tokens, excluded from attributions) must not exceed 1.
    assert 0 < sum(a.weight for a in explanation.attributions) <= 1.0 + 1e-6
    details = explanation.details
    assert details["num_layers"] == 2
    assert len(details["layers"]) == 2
    assert len(details["layers"][0]) == len(details["tokens"])


def test_attention_layer_out_of_range_raises(tmp_path) -> None:
    duck = _tiny_bert(tmp_path)
    with pytest.raises(ConfigurationError, match="layer"):
        get_explainer("attention", layer=99).explain(duck, "kaj sie owce")


def test_attention_marker_fallback_merges_wordpieces() -> None:
    # The marker-based fallback path is pure Python: exercise it directly
    # without torch or transformers installed.
    from tulip.explain.attention import _merge_by_markers

    tokens = ["[CLS]", "kaj", "ow", "##ce", "pas", "##a", "[SEP]"]
    weights = np.array([0.4, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05])
    merged = _merge_by_markers(tokens, weights, {"[CLS]", "[SEP]"})
    assert merged == [
        ("kaj", pytest.approx(0.1)),
        ("owce", pytest.approx(0.3)),
        ("pasa", pytest.approx(0.15)),
    ]


def test_attention_marker_fallback_handles_sentencepiece() -> None:
    from tulip.explain.attention import _merge_by_markers

    tokens = ["<s>", "▁kaj", "▁ow", "ce", "</s>"]
    weights = np.array([0.5, 0.2, 0.2, 0.1, 0.0])
    merged = _merge_by_markers(tokens, weights, {"<s>", "</s>"})
    assert merged == [("kaj", pytest.approx(0.2)), ("owce", pytest.approx(0.3))]
