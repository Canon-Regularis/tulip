"""Tests for model persistence (``tulip.models.persistence``)."""

from __future__ import annotations

import importlib.util
import platform
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from tulip.core.exceptions import ConfigurationError, DataError


def _stub_missing_model_siblings() -> None:
    """Allow importing ``tulip.models`` while sibling modules are unwritten.

    ``tulip/models/__init__.py`` eagerly imports every built-in model module
    for registration; during parallel development some siblings may not exist
    yet. Injecting empty placeholder modules keeps this test module runnable
    and is a no-op once the real files exist.
    """
    spec = importlib.util.find_spec("tulip")
    assert spec is not None and spec.origin is not None
    models_dir = Path(spec.origin).parent / "models"
    for sibling in ("classical", "fasttext_model", "neural_audio", "neural_text"):
        if not (models_dir / f"{sibling}.py").exists():
            name = f"tulip.models.{sibling}"
            sys.modules.setdefault(name, types.ModuleType(name))


_stub_missing_model_siblings()

from tulip.models import MODELS  # noqa: E402
from tulip.models.persistence import (  # noqa: E402
    FORMAT_VERSION,
    METADATA_FILENAME,
    MODEL_FILENAME,
    load_model,
    save_model,
)


@pytest.fixture
def fitted(
    synthetic_texts_and_labels: tuple[list[str], list[str]],
) -> tuple[object, object, np.ndarray]:
    """A fitted logistic regression plus its TF-IDF matrix and labels."""
    texts, labels = synthetic_texts_and_labels
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts)
    label_array = np.asarray(labels)
    model = MODELS.create("logistic_regression", seed=0).fit(matrix, label_array)
    return model, matrix, label_array


def test_save_creates_expected_files(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray]
) -> None:
    model, _, _ = fitted
    target = save_model(model, tmp_path / "artifact", {"experiment": "unit-test"})
    assert target == tmp_path / "artifact"
    assert (target / MODEL_FILENAME).is_file()
    assert (target / METADATA_FILENAME).is_file()


def test_roundtrip_preserves_predictions_exactly(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray]
) -> None:
    model, matrix, _ = fitted
    save_model(model, tmp_path / "artifact", {"experiment": "unit-test"})
    loaded, _ = load_model(tmp_path / "artifact")
    np.testing.assert_array_equal(loaded.predict(matrix), model.predict(matrix))
    np.testing.assert_array_equal(loaded.predict_proba(matrix), model.predict_proba(matrix))


def test_metadata_contents(tmp_path: Path, fitted: tuple[object, object, np.ndarray]) -> None:
    model, _, labels = fitted
    user_metadata = {"experiment": "unit-test", "note": "gwara podhalańska", "folds": 3}
    save_model(model, tmp_path / "artifact", user_metadata)
    _, metadata = load_model(tmp_path / "artifact")
    assert metadata["format_version"] == FORMAT_VERSION
    assert metadata["model_class"].endswith("LogisticRegression")
    assert isinstance(metadata["tulip_version"], str) and metadata["tulip_version"]
    assert metadata["python_version"] == platform.python_version()
    assert metadata["classes"] == sorted(set(labels))
    assert metadata["metadata"] == user_metadata  # UTF-8 round-trip incl. diacritics


def test_metadata_is_deterministic(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray]
) -> None:
    model, _, _ = fitted
    save_model(model, tmp_path / "first", {"metrics": {"accuracy": 0.9}})
    save_model(model, tmp_path / "second", {"metrics": {"accuracy": 0.9}})
    first = (tmp_path / "first" / METADATA_FILENAME).read_bytes()
    second = (tmp_path / "second" / METADATA_FILENAME).read_bytes()
    assert first == second  # sorted keys, no timestamps


def test_numpy_metadata_values_coerced(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray]
) -> None:
    model, _, _ = fitted
    metadata = {"accuracy": np.float64(0.875), "support": np.array([12, 12, 12, 12])}
    save_model(model, tmp_path / "artifact", metadata)
    _, loaded = load_model(tmp_path / "artifact")
    assert loaded["metadata"]["accuracy"] == pytest.approx(0.875)
    assert loaded["metadata"]["support"] == [12, 12, 12, 12]


def test_full_pipeline_roundtrip(
    tmp_path: Path, synthetic_texts_and_labels: tuple[list[str], list[str]]
) -> None:
    texts, labels = synthetic_texts_and_labels
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
            ("clf", MODELS.create("naive_bayes")),
        ]
    ).fit(texts, labels)
    save_model(pipeline, tmp_path / "pipeline", {"kind": "text-baseline"})
    loaded, metadata = load_model(tmp_path / "pipeline")
    np.testing.assert_array_equal(loaded.predict(texts), pipeline.predict(texts))
    assert metadata["classes"] == sorted(set(labels))


def test_unfitted_model_saves_with_null_classes(tmp_path: Path) -> None:
    save_model(MODELS.create("naive_bayes"), tmp_path / "unfitted", {})
    _, metadata = load_model(tmp_path / "unfitted")
    assert metadata["classes"] is None


def test_load_missing_path_raises_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_model(tmp_path / "does-not-exist")


@pytest.mark.parametrize("missing_file", [MODEL_FILENAME, METADATA_FILENAME])
def test_load_incomplete_artifact_raises(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray], missing_file: str
) -> None:
    model, _, _ = fitted
    target = save_model(model, tmp_path / "artifact", {})
    (target / missing_file).unlink()
    with pytest.raises(DataError, match=missing_file):
        load_model(target)


def test_load_corrupt_metadata_raises(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray]
) -> None:
    model, _, _ = fitted
    target = save_model(model, tmp_path / "artifact", {})
    (target / METADATA_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(DataError, match="corrupt"):
        load_model(target)


def test_load_non_object_metadata_raises(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray]
) -> None:
    model, _, _ = fitted
    target = save_model(model, tmp_path / "artifact", {})
    (target / METADATA_FILENAME).write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(DataError, match="JSON object"):
        load_model(target)


def test_load_corrupt_model_raises(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray]
) -> None:
    model, _, _ = fitted
    target = save_model(model, tmp_path / "artifact", {})
    (target / MODEL_FILENAME).write_bytes(b"definitely not a joblib payload")
    with pytest.raises(DataError, match="corrupt"):
        load_model(target)


def test_non_serialisable_metadata_fails_before_writing(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray]
) -> None:
    model, _, _ = fitted
    with pytest.raises(ConfigurationError, match="JSON"):
        save_model(model, tmp_path / "artifact", {"bad": object()})
    assert not (tmp_path / "artifact").exists()  # no partial artifact left behind


def test_save_over_existing_file_raises(
    tmp_path: Path, fitted: tuple[object, object, np.ndarray]
) -> None:
    model, _, _ = fitted
    blocker = tmp_path / "blocker"
    blocker.write_text("occupied", encoding="utf-8")
    with pytest.raises(DataError, match="not a directory"):
        save_model(model, blocker, {})
