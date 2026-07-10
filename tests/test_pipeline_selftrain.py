"""Tests for semi-supervised self-training (pseudo-labeling).

Hermetic: the tiny synthetic corpus from :mod:`conftest` stands in for a
labeled seed set, and label-stripped copies stand in for an unlabeled corpus
such as ``bigos``. ``logistic_regression`` is used throughout because its
softmax confidences on this corpus stay comfortably below 1.0 (~0.79 max),
which makes the "impossibly high threshold" case provably admit nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_samples
from tulip.core.exceptions import DataError
from tulip.core.types import DialectLabels, Prediction, Sample
from tulip.labels.taxonomy import LabelLevel
from tulip.pipeline.selftrain import SelfTrainConfig, SelfTrainResult, self_train

FEATURES = ["char_tfidf", "word_tfidf"]
PODHALE_QUERY = "Hej, baca się pyto, kaj się owce pasą na holi."


def _unlabeled_from(labeled: list[Sample]) -> list[Sample]:
    """Label-stripped copies of ``labeled`` standing in for an unlabeled corpus."""
    return [
        Sample(
            id=f"unl-{sample.id}",
            text=sample.text,
            speaker_id=sample.speaker_id,
            source="unlabeled",
        )
        for sample in labeled
    ]


def _run(config: SelfTrainConfig) -> SelfTrainResult:
    """Self-train over the synthetic seed set and its label-stripped copies."""
    labeled = make_samples(repeats=3)
    return self_train(
        labeled=labeled,
        unlabeled=_unlabeled_from(labeled),
        model="logistic_regression",
        features=FEATURES,
        config=config,
    )


class TestPseudoLabeling:
    def test_pseudo_samples_added_with_sane_counts(self) -> None:
        result = _run(SelfTrainConfig(confidence_threshold=0.5, max_iterations=3))

        assert result.pseudo_samples, "expected some confident pseudo-labels"
        # Early stopping guarantees no zero-count round is ever recorded.
        assert all(count > 0 for count in result.n_pseudo_per_iteration)
        assert result.iterations == len(result.n_pseudo_per_iteration)
        # Per-round counts sum to the pseudo-sample total; running total rises.
        assert sum(result.n_pseudo_per_iteration) == len(result.pseudo_samples)
        running = [
            sum(result.n_pseudo_per_iteration[: i + 1])
            for i in range(len(result.n_pseudo_per_iteration))
        ]
        assert running == sorted(running)
        # Each unlabeled sample is pseudo-labeled at most once.
        pseudo_ids = [s.id for s in result.pseudo_samples]
        assert len(pseudo_ids) == len(set(pseudo_ids))
        assert len(result.pseudo_samples) <= len(_unlabeled_from(make_samples(repeats=3)))

    def test_max_pseudo_per_iter_caps_each_round(self) -> None:
        result = _run(
            SelfTrainConfig(confidence_threshold=0.5, max_pseudo_per_iter=5, max_iterations=3)
        )
        assert result.pseudo_samples
        assert all(count <= 5 for count in result.n_pseudo_per_iteration)

    def test_pseudo_samples_carry_label_and_provenance(self) -> None:
        result = _run(SelfTrainConfig(confidence_threshold=0.5, max_iterations=2))

        assert result.pseudo_samples
        for pseudo in result.pseudo_samples:
            assert pseudo.metadata["pseudo_labeled"] is True
            assert isinstance(pseudo.metadata["confidence"], float)
            # Target-level label was assigned from the prediction.
            label = pseudo.labels.at_level(LabelLevel.DIALECT)
            assert label in result.classifier.classes_


class TestConvergence:
    def test_impossible_threshold_adds_nothing_and_stops_early(self) -> None:
        # Softmax confidences on this corpus peak near 0.79, so a 0.999 bar is
        # unreachable: the very first round must converge with zero additions.
        result = _run(SelfTrainConfig(confidence_threshold=0.999, max_iterations=3))

        assert result.pseudo_samples == ()
        assert result.n_pseudo_per_iteration == ()
        assert result.iterations == 0
        # The final classifier is still the fitted seed model.
        assert result.classifier.predict(PODHALE_QUERY).label in result.classifier.classes_

    def test_no_usable_unlabeled_is_a_noop(self) -> None:
        labeled = make_samples(repeats=3)
        # Audio-only unlabeled samples carry no text, so a text task ignores them.
        unlabeled = [
            Sample(id="a1", audio_path=Path("a1.wav"), speaker_id="spk"),
            Sample(id="a2", audio_path=Path("a2.wav"), speaker_id="spk"),
        ]
        result = self_train(
            labeled=labeled,
            unlabeled=unlabeled,
            model="logistic_regression",
            features=FEATURES,
            config=SelfTrainConfig(confidence_threshold=0.5),
        )
        assert result.pseudo_samples == ()
        assert result.iterations == 0


class TestDeterminism:
    def test_identical_runs_match(self) -> None:
        config = SelfTrainConfig(confidence_threshold=0.5, max_iterations=3, seed=42)
        first = _run(config)
        second = _run(config)

        assert first.n_pseudo_per_iteration == second.n_pseudo_per_iteration
        assert [s.id for s in first.pseudo_samples] == [s.id for s in second.pseudo_samples]

        queries = [PODHALE_QUERY, "Jo żech je z Katowic i godom po naszymu."]
        first_preds = first.classifier.predict_batch(queries)
        second_preds = second.classifier.predict_batch(queries)
        assert [p.label for p in first_preds] == [p.label for p in second_preds]
        for a, b in zip(first_preds, second_preds, strict=True):
            assert a.as_dict() == pytest.approx(b.as_dict())


class TestPurity:
    def test_original_samples_are_not_mutated(self) -> None:
        labeled = make_samples(repeats=3)
        unlabeled = _unlabeled_from(labeled)
        labeled_snapshot = [s.model_dump() for s in labeled]
        unlabeled_snapshot = [s.model_dump() for s in unlabeled]

        self_train(
            labeled=labeled,
            unlabeled=unlabeled,
            model="logistic_regression",
            features=FEATURES,
            config=SelfTrainConfig(confidence_threshold=0.5, max_iterations=3),
        )

        assert [s.model_dump() for s in labeled] == labeled_snapshot
        assert [s.model_dump() for s in unlabeled] == unlabeled_snapshot
        # No pseudo provenance leaked onto the original unlabeled inputs.
        for sample in unlabeled:
            assert "pseudo_labeled" not in sample.metadata
            assert sample.labels.at_level(LabelLevel.DIALECT) is None


class TestFinalClassifier:
    def test_final_classifier_is_fitted_and_predicts(self) -> None:
        result = _run(SelfTrainConfig(confidence_threshold=0.5, max_iterations=2))
        prediction = result.classifier.predict(PODHALE_QUERY)
        assert isinstance(prediction, Prediction)
        assert prediction.label == "podhale"

    def test_config_defaults_apply_when_omitted(self) -> None:
        labeled = make_samples(repeats=3)
        result = self_train(
            labeled=labeled,
            unlabeled=_unlabeled_from(labeled),
            model="logistic_regression",
            features=FEATURES,
        )
        # Default 0.90 threshold is above the corpus's ~0.79 peak: no additions.
        assert result.n_pseudo_per_iteration == ()
        assert result.classifier.predict(PODHALE_QUERY).label == "podhale"


class TestConfigValidation:
    def test_confidence_threshold_must_be_a_probability(self) -> None:
        with pytest.raises(ValueError):
            SelfTrainConfig(confidence_threshold=1.5)

    def test_max_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            SelfTrainConfig(max_iterations=0)

    def test_empty_labeled_seed_raises(self) -> None:
        with pytest.raises(DataError, match="labeled seed"):
            self_train(
                labeled=[],
                unlabeled=[Sample(id="u1", text="cokolwiek", speaker_id="s")],
                model="logistic_regression",
                features=FEATURES,
            )


def test_pseudo_family_target_derives_family() -> None:
    """A FAMILY-target run assigns family labels (auto-derived for dialects)."""
    labeled = make_samples(repeats=3)
    result = self_train(
        labeled=labeled,
        unlabeled=_unlabeled_from(labeled),
        model="logistic_regression",
        features=FEATURES,
        config=SelfTrainConfig(confidence_threshold=0.5, target=LabelLevel.FAMILY),
    )
    assert result.pseudo_samples
    families = result.classifier.classes_
    for pseudo in result.pseudo_samples:
        assert pseudo.labels.at_level(LabelLevel.FAMILY) in families
    # Sanity: DialectLabels still constructs cleanly for a family label.
    assert DialectLabels(family="standard").family == "standard"
