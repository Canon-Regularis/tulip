"""Tests for multimodal text + audio late fusion (:mod:`tulip.pipeline.fusion`).

Hermetic throughout. The fusion strategies are pure numpy and tested directly on
hand-built stacks. The composite classifier is exercised with a deterministic
:class:`StubClassifier` standing in for an audio expert (training a real speech
model is neither hermetic nor fast) paired, where the task calls for it, with a
real text :class:`~tulip.pipeline.classifier.DialectClassifier` trained on the
synthetic corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import Prediction, Sample, TaskType
from tulip.data.synthetic import SyntheticSpec, generate_corpus
from tulip.labels.taxonomy import LabelLevel
from tulip.pipeline.classifier import DialectClassifier
from tulip.pipeline.fusion import (
    FusionStrategy,
    LogarithmicPoolingFusion,
    MaximumFusion,
    MultimodalClassifier,
    WeightedAverageFusion,
    build_strategy,
)
from tulip.pipeline.protocols import SamplePredictor

if TYPE_CHECKING:
    from collections.abc import Sequence


class StubClassifier:
    """A deterministic :class:`ProbabilisticClassifier` for hermetic fusion tests.

    Returns the same distribution for every input, so tests can assert exact
    fused values without training anything. Exposes exactly the structural
    surface fusion depends on (``classes_``, ``target``, ``task``,
    ``predict_proba``) and, deliberately, no ``fit``/``save`` -- proving the
    composite treats it purely through the protocol.
    """

    def __init__(
        self,
        *,
        classes: Sequence[str],
        task: TaskType,
        target: LabelLevel = LabelLevel.DIALECT,
        distribution: np.ndarray | None = None,
    ) -> None:
        self.classes_ = tuple(classes)
        self.task = task
        self.target = target
        if distribution is None:
            distribution = np.full(len(self.classes_), 1.0 / len(self.classes_))
        self._distribution = np.asarray(distribution, dtype=np.float64)

    def predict_proba(self, raws: Sequence[Any]) -> np.ndarray:
        return np.tile(self._distribution, (len(list(raws)), 1))


STRATEGIES: list[FusionStrategy] = [
    WeightedAverageFusion((0.5, 0.5)),
    MaximumFusion(),
    LogarithmicPoolingFusion((0.5, 0.5)),
]
STRATEGY_IDS = ["weighted_average", "maximum", "logarithmic_pooling"]


def _stack_and_mask() -> tuple[np.ndarray, np.ndarray]:
    """A (2 modalities, 4 samples, 3 classes) stack with varied modality presence.

    Sample layout: 0 both present, 1 text-only, 2 audio-only, 3 both present.
    Each modality-sample row is a proper distribution (Dirichlet draw).
    """
    rng = np.random.default_rng(11)
    stack = rng.dirichlet(np.ones(3), size=(2, 4))
    mask = np.array(
        [
            [True, True, False, True],
            [True, False, True, True],
        ],
        dtype=bool,
    )
    return stack, mask


def _text_corpus() -> list[Sample]:
    """A small, cleanly separable synthetic text corpus (all dialect-labelled)."""
    spec = SyntheticSpec(
        n_speakers_per_dialect=3,
        samples_per_speaker=4,
        dialects=("podhale", "silesia", "kurpie"),
        include_standard=False,
        noise_level=0.0,
        marker_dropout=0.0,
        seed=7,
    )
    return generate_corpus(spec)


def _text_classifier(corpus: list[Sample]) -> DialectClassifier:
    """A real, fitted text DialectClassifier over ``corpus``."""
    return DialectClassifier(
        model="logistic_regression",
        features=["char_tfidf"],
        task=TaskType.TEXT,
        target=LabelLevel.DIALECT,
        seed=42,
    ).fit(corpus)


# --------------------------------------------------------------- strategy contract


@pytest.mark.parametrize("strategy", STRATEGIES, ids=STRATEGY_IDS)
class TestFusionStrategyContract:
    """The identical postcondition every FusionStrategy must satisfy (LSP)."""

    def test_output_shape_and_rows_sum_to_one(self, strategy: FusionStrategy) -> None:
        stack, mask = _stack_and_mask()
        fused = strategy.fuse(stack, mask)
        assert fused.shape == (4, 3)
        np.testing.assert_allclose(fused.sum(axis=1), 1.0)

    def test_no_nan(self, strategy: FusionStrategy) -> None:
        stack, mask = _stack_and_mask()
        fused = strategy.fuse(stack, mask)
        assert not np.isnan(fused).any()

    def test_lone_present_modality_passes_through_unchanged(self, strategy: FusionStrategy) -> None:
        stack, mask = _stack_and_mask()
        fused = strategy.fuse(stack, mask)
        # Sample 1 is text-only (modality 0); sample 2 is audio-only (modality 1).
        np.testing.assert_allclose(fused[1], stack[0, 1])
        np.testing.assert_allclose(fused[2], stack[1, 2])

    def test_no_modality_present_raises(self, strategy: FusionStrategy) -> None:
        stack, mask = _stack_and_mask()
        mask = mask.copy()
        mask[:, 0] = False  # sample 0 now has no present modality
        with pytest.raises(DataError):
            strategy.fuse(stack, mask)

    def test_a_degenerate_zero_sum_row_falls_back_to_uniform(
        self, strategy: FusionStrategy
    ) -> None:
        """A present modality whose row is all zeros must not renormalise to NaN.

        The strategy is a public value object; a caller can hand it a stack whose
        pooled row sums to zero. The no-NaN postcondition still holds -- such a
        row becomes uniform.
        """
        stack = np.zeros((2, 1, 3))  # one sample; present modality row is all-zero
        mask = np.array([[True], [False]])
        fused = strategy.fuse(stack, mask)
        assert not np.isnan(fused).any()
        np.testing.assert_allclose(fused.sum(axis=1), 1.0)
        np.testing.assert_allclose(fused[0], np.full(3, 1 / 3))

    def test_satisfies_fusion_strategy_protocol(self, strategy: FusionStrategy) -> None:
        assert isinstance(strategy, FusionStrategy)


# ----------------------------------------------------------- per-strategy semantics


def test_weighted_average_is_convex_combination() -> None:
    stack = np.array([[[0.6, 0.4]], [[0.2, 0.8]]])
    mask = np.ones((2, 1), dtype=bool)
    fused = WeightedAverageFusion((0.25, 0.75)).fuse(stack, mask)
    expected = 0.25 * stack[0, 0] + 0.75 * stack[1, 0]
    np.testing.assert_allclose(fused[0], expected / expected.sum())


def test_weighted_average_all_weight_on_text_reproduces_text_exactly() -> None:
    rng = np.random.default_rng(3)
    stack = rng.dirichlet(np.ones(4), size=(2, 5))
    mask = np.ones((2, 5), dtype=bool)  # both modalities present everywhere
    fused = WeightedAverageFusion((1.0, 0.0)).fuse(stack, mask)
    np.testing.assert_allclose(fused, stack[0])


def test_maximum_takes_elementwise_max_then_renormalises() -> None:
    stack = np.array([[[0.8, 0.1, 0.1]], [[0.2, 0.7, 0.1]]])
    mask = np.ones((2, 1), dtype=bool)
    fused = MaximumFusion().fuse(stack, mask)
    expected = np.array([0.8, 0.7, 0.1])
    np.testing.assert_allclose(fused[0], expected / expected.sum())


def test_logarithmic_pooling_is_weighted_geometric_mean() -> None:
    stack = np.array([[[0.9, 0.1]], [[0.2, 0.8]]])
    mask = np.ones((2, 1), dtype=bool)
    fused = LogarithmicPoolingFusion((0.5, 0.5)).fuse(stack, mask)
    geometric = np.sqrt(stack[0, 0] * stack[1, 0])  # weights (0.5, 0.5)
    np.testing.assert_allclose(fused[0], geometric / geometric.sum())


# ---------------------------------------------------------------- weight validation


def test_weighted_average_rejects_all_zero_weights() -> None:
    with pytest.raises(ConfigurationError, match="positive weight"):
        WeightedAverageFusion((0.0, 0.0))


def test_logarithmic_pooling_rejects_negative_weight() -> None:
    with pytest.raises(ConfigurationError, match="non-negative"):
        LogarithmicPoolingFusion((-0.1, 1.0))


# --------------------------------------------------------------- build_strategy


def test_build_strategy_round_trips_each_kind() -> None:
    strategy = build_strategy("weighted_average", {"weights": [0.3, 0.7]})
    assert isinstance(strategy, WeightedAverageFusion)
    assert strategy.weights == (0.3, 0.7)
    assert isinstance(build_strategy("maximum"), MaximumFusion)
    assert isinstance(
        build_strategy("logarithmic_pooling", {"weights": [1.0, 2.0]}), LogarithmicPoolingFusion
    )


def test_build_strategy_unknown_kind_raises() -> None:
    with pytest.raises(ConfigurationError, match="unknown fusion strategy"):
        build_strategy("nonesuch")


def test_build_strategy_missing_param_raises() -> None:
    with pytest.raises(ConfigurationError, match="missing required parameter"):
        build_strategy("weighted_average", {})


# ------------------------------------------------------- MultimodalClassifier: config


def test_target_mismatch_raises() -> None:
    text = StubClassifier(classes=("a", "b"), task=TaskType.TEXT, target=LabelLevel.DIALECT)
    audio = StubClassifier(classes=("a", "b"), task=TaskType.AUDIO, target=LabelLevel.FAMILY)
    with pytest.raises(ConfigurationError, match="target"):
        MultimodalClassifier(text=text, audio=audio)


def test_wrong_text_task_raises() -> None:
    text = StubClassifier(classes=("a", "b"), task=TaskType.AUDIO)  # should be TEXT
    audio = StubClassifier(classes=("a", "b"), task=TaskType.AUDIO)
    with pytest.raises(ConfigurationError, match="text base"):
        MultimodalClassifier(text=text, audio=audio)


def test_wrong_audio_task_raises() -> None:
    text = StubClassifier(classes=("a", "b"), task=TaskType.TEXT)
    audio = StubClassifier(classes=("a", "b"), task=TaskType.TEXT)  # should be AUDIO
    with pytest.raises(ConfigurationError, match="audio base"):
        MultimodalClassifier(text=text, audio=audio)


def test_is_sample_predictor() -> None:
    text = StubClassifier(classes=("a", "b"), task=TaskType.TEXT)
    audio = StubClassifier(classes=("a", "b"), task=TaskType.AUDIO)
    fused = MultimodalClassifier(text=text, audio=audio)
    assert isinstance(fused, SamplePredictor)


# ---------------------------------------------------- MultimodalClassifier: fusion


def test_missing_audio_passes_text_distribution_through() -> None:
    text = StubClassifier(
        classes=("a", "b", "c"), task=TaskType.TEXT, distribution=np.array([0.6, 0.3, 0.1])
    )
    audio = StubClassifier(
        classes=("a", "b", "c"), task=TaskType.AUDIO, distribution=np.array([0.1, 0.1, 0.8])
    )
    fused = MultimodalClassifier(text=text, audio=audio)  # default equal weights
    sample = Sample(id="t1", text="jakiś tekst", speaker_id="spk")  # no audio_path
    proba = fused.predict_proba_samples([sample])
    np.testing.assert_allclose(proba[0], [0.6, 0.3, 0.1])


def test_class_union_alignment_zeroes_unknown_class() -> None:
    # Audio knows only 2 of the 3 classes; with all weight on audio it must
    # contribute exactly 0 to the class it never saw.
    text = StubClassifier(
        classes=("a", "b", "c"), task=TaskType.TEXT, distribution=np.array([0.2, 0.3, 0.5])
    )
    audio = StubClassifier(
        classes=("a", "b"), task=TaskType.AUDIO, distribution=np.array([0.4, 0.6])
    )
    fused = MultimodalClassifier(text=text, audio=audio, strategy=WeightedAverageFusion((0.0, 1.0)))
    assert fused.classes_ == ("a", "b", "c")
    sample = Sample(id="s1", text="tekst", audio_path=Path("s1.wav"), speaker_id="spk")
    proba = fused.predict_proba_samples([sample])
    assert proba[0, fused.classes_.index("c")] == 0.0
    np.testing.assert_allclose(proba[0], [0.4, 0.6, 0.0])
    assert not np.isnan(proba).any()


def test_sample_with_no_modality_raises_naming_it() -> None:
    text = StubClassifier(classes=("a", "b"), task=TaskType.TEXT)
    audio = StubClassifier(classes=("a", "b"), task=TaskType.AUDIO)
    fused = MultimodalClassifier(text=text, audio=audio)
    # Sample enforces at-least-one-modality; bypass it to reach the fusion guard.
    empty = Sample.model_construct(id="empty-42", text=None, audio_path=None)
    with pytest.raises(DataError, match="empty-42"):
        fused.predict_proba_samples([empty])


def test_empty_sample_list_returns_empty_matrix() -> None:
    text = StubClassifier(classes=("a", "b"), task=TaskType.TEXT)
    audio = StubClassifier(classes=("b", "c"), task=TaskType.AUDIO)
    fused = MultimodalClassifier(text=text, audio=audio)
    proba = fused.predict_proba_samples([])
    assert proba.shape == (0, 3)  # union {a, b, c}


# ---------------------------------------------------- MultimodalClassifier: fit


def test_fit_trains_only_real_bases_and_realigns_classes() -> None:
    corpus = _text_corpus()
    text = DialectClassifier(
        model="logistic_regression",
        features=["char_tfidf"],
        task=TaskType.TEXT,
        target=LabelLevel.DIALECT,
        seed=42,
    )
    audio = StubClassifier(classes=("podhale",), task=TaskType.AUDIO, target=LabelLevel.DIALECT)
    fused = MultimodalClassifier(text=text, audio=audio)
    # Before fit the text base is unfitted (classes_ == ()); union is the stub's.
    assert fused.classes_ == ("podhale",)

    returned = fused.fit(corpus)
    assert returned is fused
    # The text base learned the corpus dialects; the union grew to include them.
    assert text.classes_  # now fitted
    assert set(text.classes_) <= set(fused.classes_)
    assert "podhale" in fused.classes_
    assert len(fused.classes_) > 1


# ---------------------------------------------------- MultimodalClassifier: end-to-end


def test_end_to_end_real_text_and_stub_audio_yields_valid_predictions() -> None:
    corpus = _text_corpus()
    text = _text_classifier(corpus)
    audio = StubClassifier(classes=text.classes_, task=TaskType.AUDIO, target=LabelLevel.DIALECT)
    fused = MultimodalClassifier(text=text, audio=audio)
    assert fused.classes_ == tuple(sorted(text.classes_))

    queries = [
        Sample(id=f"q{i}", text=sample.text, audio_path=Path(f"q{i}.wav"), speaker_id="spk")
        for i, sample in enumerate(corpus[:6])
    ]
    predictions = fused.predict_samples(queries)

    assert len(predictions) == len(queries)
    for prediction in predictions:
        assert isinstance(prediction, Prediction)
        assert prediction.level == LabelLevel.DIALECT
        assert prediction.label in fused.classes_
        assert len(prediction.probabilities) == len(fused.classes_)
        assert sum(cp.probability for cp in prediction.probabilities) == pytest.approx(1.0)


def test_predictions_are_deterministic() -> None:
    corpus = _text_corpus()

    def run() -> np.ndarray:
        text = _text_classifier(corpus)
        audio = StubClassifier(
            classes=text.classes_, task=TaskType.AUDIO, target=LabelLevel.DIALECT
        )
        fused = MultimodalClassifier(
            text=text, audio=audio, strategy=LogarithmicPoolingFusion((0.5, 0.5))
        )
        queries = [
            Sample(id=f"q{i}", text=sample.text, audio_path=Path(f"q{i}.wav"), speaker_id="spk")
            for i, sample in enumerate(corpus[:5])
        ]
        return fused.predict_proba_samples(queries)

    np.testing.assert_array_equal(run(), run())


# ---------------------------------------------------- MultimodalClassifier: persistence


def test_save_load_round_trip(tmp_path: Path) -> None:
    corpus = _text_corpus()
    text = _text_classifier(corpus)
    # A persistable audio base: a text-trained classifier relabelled AUDIO. Its
    # (text) pipeline is never invoked below because the queries carry no audio,
    # so a lone text modality passes through -- we only need a *saveable*
    # DialectClassifier with task=AUDIO to exercise the composite artifact.
    audio = _text_classifier(corpus)
    audio.task = TaskType.AUDIO
    fused = MultimodalClassifier(
        text=text, audio=audio, strategy=LogarithmicPoolingFusion((0.4, 0.6))
    )
    queries = [
        Sample(id=f"q{i}", text=sample.text, speaker_id="spk")
        for i, sample in enumerate(corpus[:5])
    ]
    before = fused.predict_samples(queries)

    artifact = tmp_path / "mm"
    fused.save(artifact)
    loaded = MultimodalClassifier.load(artifact)

    assert loaded.target == LabelLevel.DIALECT
    assert isinstance(loaded.strategy, LogarithmicPoolingFusion)
    assert loaded.strategy.weights == (0.4, 0.6)
    assert loaded.classes_ == fused.classes_

    after = loaded.predict_samples(queries)
    assert [p.label for p in after] == [p.label for p in before]
    for restored, original in zip(after, before, strict=True):
        assert restored.as_dict() == pytest.approx(original.as_dict())


def test_save_rejects_unpersistable_base(tmp_path: Path) -> None:
    text = StubClassifier(classes=("a", "b"), task=TaskType.TEXT)
    audio = StubClassifier(classes=("a", "b"), task=TaskType.AUDIO)
    fused = MultimodalClassifier(text=text, audio=audio)
    with pytest.raises(ConfigurationError, match="cannot be persisted"):
        fused.save(tmp_path / "mm")


def test_load_missing_sidecar_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataError, match=r"missing fusion\.json"):
        MultimodalClassifier.load(empty)


def test_load_wrong_kind_raises(tmp_path: Path) -> None:
    artifact = tmp_path / "mm"
    artifact.mkdir()
    (artifact / "fusion.json").write_text('{"kind": "SomethingElse"}', encoding="utf-8")
    with pytest.raises(DataError, match="not saved by MultimodalClassifier"):
        MultimodalClassifier.load(artifact)
