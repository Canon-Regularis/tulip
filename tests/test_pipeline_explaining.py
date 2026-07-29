"""Tests for tulip.pipeline.explaining.PredictionExplainer routing.

Covers the target-object selection (_final_estimator), the lazily-built and
cached nearest-examples index, and the retrieval-transformer paths for both
sklearn pipelines and raw-input text/audio models.
"""

from __future__ import annotations

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import DialectLabels, Sample, TaskType
from tulip.pipeline.explaining import PredictionExplainer

_MARKERS = {"podhale": "baca hej kaj hala", "silesia": "gynau tref ja ci", "kurpie": "kie psa wej"}


def _samples() -> list[Sample]:
    samples: list[Sample] = []
    n = 0
    for dialect, text in _MARKERS.items():
        for i in range(3):
            n += 1
            samples.append(
                Sample(
                    id=f"s{n}",
                    text=f"{text} {i}",
                    speaker_id=f"{dialect[:3]}{i}",
                    labels=DialectLabels(dialect=dialect),
                )
            )
    return samples


def _fitted_pipeline(samples: list[Sample]) -> Pipeline:
    texts = [s.text or "" for s in samples]
    labels = [s.labels.dialect for s in samples]
    return Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=2000, random_state=0))]
    ).fit(texts, labels)


class _RawTextModel:
    """A raw-input model (not a sklearn Pipeline) that takes texts directly."""

    task = TaskType.TEXT


# ------------------------------------------------------------- _final_estimator


def test_final_estimator_returns_the_last_pipeline_step() -> None:
    samples = _samples()
    pipeline = _fitted_pipeline(samples)
    explainer = PredictionExplainer(pipeline=pipeline, task=TaskType.TEXT, train_samples=samples)
    assert explainer._final_estimator() is pipeline.steps[-1][1]


def test_final_estimator_returns_a_raw_model_unchanged() -> None:
    raw = _RawTextModel()
    explainer = PredictionExplainer(pipeline=raw, task=TaskType.TEXT)
    assert explainer._final_estimator() is raw


# ----------------------------------------------------------- nearest_examples


def test_nearest_examples_over_a_pipeline_is_built_once_and_cached() -> None:
    samples = _samples()
    explainer = PredictionExplainer(
        pipeline=_fitted_pipeline(samples), task=TaskType.TEXT, train_samples=samples
    )
    first = explainer.explain(samples[0].text, method="nearest_examples")
    assert first.method == "nearest_examples"
    assert first.neighbors
    # The index is cached: a second call returns the same explainer instance's index.
    index = explainer._neighbor_explainer
    explainer.explain(samples[1].text, method="nearest_examples")
    assert explainer._neighbor_explainer is index


def test_nearest_examples_for_a_raw_text_model_fits_a_char_tfidf() -> None:
    samples = _samples()
    explainer = PredictionExplainer(
        pipeline=_RawTextModel(), task=TaskType.TEXT, train_samples=samples
    )
    explanation = explainer.explain(samples[0].text, method="nearest_examples")
    assert explanation.neighbors
    # The most similar indexed sample to a query equal to a training text is itself.
    assert explanation.neighbors[0].sample_id == samples[0].id


def test_nearest_examples_without_training_samples_raises() -> None:
    explainer = PredictionExplainer(pipeline=_RawTextModel(), task=TaskType.TEXT, train_samples=())
    with pytest.raises(ConfigurationError, match="training samples"):
        explainer.explain("baca hej", method="nearest_examples")


def test_nearest_examples_for_a_raw_audio_model_raises() -> None:
    samples = _samples()

    class _RawAudioModel:
        task = TaskType.AUDIO

    explainer = PredictionExplainer(
        pipeline=_RawAudioModel(), task=TaskType.AUDIO, train_samples=samples
    )
    with pytest.raises(ConfigurationError, match="text-based"):
        explainer.explain("baca hej", method="nearest_examples")


# ------------------------------------------------------------------- top_tfidf


def test_top_tfidf_routes_to_the_full_pipeline() -> None:
    samples = _samples()
    explainer = PredictionExplainer(
        pipeline=_fitted_pipeline(samples), task=TaskType.TEXT, train_samples=samples
    )
    explanation = explainer.explain(samples[0].text, method="top_tfidf")
    assert explanation.method == "top_tfidf"
    assert explanation.attributions
