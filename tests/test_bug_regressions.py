"""Regressions for the bugs found in the codebase-wide hunt.

One test per fixed defect, each named for the failure it guards against. The
audio wav2vec2 pooling and the manifest/serve/error-report fixes are exercised by
their own subsystem suites and are not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError, DataError


def test_label_encoded_classifier_is_a_real_sklearn_estimator() -> None:
    # Wrapping an integer-label estimator must not lose the sklearn tags: without
    # them, Pipeline.predict and is_classifier crash on sklearn >= 1.6, disabling
    # the xgboost/lightgbm baselines and their use as ensemble members.
    from sklearn.base import is_classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from tulip.models.classical import LabelEncodedClassifier

    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["a", "a", "b", "b"])
    wrapped = LabelEncodedClassifier(LogisticRegression())
    assert is_classifier(wrapped)
    pipeline = Pipeline([("model", wrapped)]).fit(x, y)
    assert list(pipeline.predict(x)) == ["a", "a", "b", "b"]
    assert pipeline.predict_proba(x).shape == (4, 2)


def test_malformed_lexicon_yaml_raises_configuration_error(tmp_path) -> None:
    from tulip.features.text.keywords import load_lexicon

    bad = tmp_path / "bad.yaml"
    bad.write_text("podhale: [baca, kaj\nsilesia: broken", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_lexicon(bad)


def test_lexicon_duplicate_key_after_casefold_raises(tmp_path) -> None:
    from tulip.features.text.keywords import load_lexicon

    dup = tmp_path / "dup.yaml"
    dup.write_text("Podhale: [baca]\npodhale: [kaj]\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate"):
        load_lexicon(dup)


def test_end_of_word_bpe_pieces_merge_into_words() -> None:
    from tulip.explain.attention import _merge_by_markers

    tokens = ["low", "er</w>", "dog</w>"]
    weights = np.array([0.5, 0.3, 0.2])
    merged = _merge_by_markers(tokens, weights, special_tokens=set())
    assert [word for word, _ in merged] == ["lower", "dog"]
    assert merged[0][1] == pytest.approx(0.8)


def test_split_fingerprint_load_reports_dataerror_on_a_bad_lock(tmp_path) -> None:
    from tulip.data.fingerprint import SplitFingerprint

    corrupt = tmp_path / "lock.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DataError):
        SplitFingerprint.load(corrupt)

    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text('{"combined": "abc", "digests": "not a mapping"}', encoding="utf-8")
    with pytest.raises(DataError):
        SplitFingerprint.load(wrong_shape)


def test_asr_multichar_variant_is_title_cased_not_upper() -> None:
    from tulip.robustness.perturbations import _substitute

    # A capitalised source with a multi-character variant becomes "Rz", not "RZ".
    assert _substitute("Z", 1.0, np.random.default_rng(0), {"z": ("rz",)}) == "Rz"
    assert _substitute("z", 1.0, np.random.default_rng(0), {"z": ("rz",)}) == "rz"


def test_project_embeddings_rejects_ragged_input_with_dataerror() -> None:
    from tulip.viz.embedding_space import _as_dense_2d

    with pytest.raises(DataError):
        _as_dense_2d([[1.0, 2.0], [3.0]])


def test_cvconfig_rejects_empty_seeds() -> None:
    from pydantic import ValidationError

    from tulip.pipeline import CVConfig

    with pytest.raises(ValidationError):
        CVConfig(seeds=())
