"""Tests for the hierarchical family->dialect backoff classifier.

Hermetic: the linguistically-grounded synthetic corpus from
:mod:`tulip.data.synthetic` supplies samples that carry *both* a dialect and a
(derived) family label, which is exactly the ground truth a coarse/fine
hierarchy needs. ``logistic_regression`` is used throughout because its softmax
confidences on this corpus peak near 0.80 (dialect) and 0.75 (family), so a
0.999 threshold is provably unreachable and forces a full backoff.
"""

from __future__ import annotations

import pytest

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import ClassProbability, Prediction
from tulip.data.synthetic import SyntheticSpec, generate_corpus
from tulip.labels.taxonomy import DialectFamily, LabelLevel, family_for
from tulip.pipeline.classifier import DialectClassifier
from tulip.pipeline.hierarchical import (
    AllOf,
    AlwaysAccept,
    AnyOf,
    ConfidenceThreshold,
    HierarchicalDialectClassifier,
    MarginThreshold,
    NotAbstained,
    PolicySpec,
    policy_from_spec,
)
from tulip.pipeline.protocols import SamplePredictor

FEATURES = ["char_tfidf", "word_tfidf"]
LEVELS = (LabelLevel.FAMILY, LabelLevel.DIALECT)
FAMILY_VALUES = {family.value for family in DialectFamily}


@pytest.fixture(scope="module")
def corpus() -> list:
    """A small corpus spanning three families, one of them multi-dialect.

    ``kurpie`` and ``masovia`` both belong to the Masovian family, so masking to
    that family keeps more than one dialect class -- the non-degenerate case.
    """
    spec = SyntheticSpec(
        n_speakers_per_dialect=2,
        samples_per_speaker=6,
        dialects=("kurpie", "masovia", "podhale", "silesia"),
        include_standard=True,
        seed=11,
    )
    return generate_corpus(spec)


def _fit(
    corpus: list,
    *,
    policy: object | None = None,
    mask_to_coarse: bool = True,
    seed: int = 42,
) -> HierarchicalDialectClassifier:
    classifier = HierarchicalDialectClassifier(
        levels=LEVELS,
        model="logistic_regression",
        features=FEATURES,
        policy=policy,  # type: ignore[arg-type]
        mask_to_coarse=mask_to_coarse,
        seed=seed,
    )
    return classifier.fit(corpus)


@pytest.fixture(scope="module")
def fitted_always(corpus: list) -> HierarchicalDialectClassifier:
    """AlwaysAccept + projection: finest level wherever the family has dialects."""
    return _fit(corpus, policy=AlwaysAccept(), mask_to_coarse=True)


@pytest.fixture(scope="module")
def fitted_always_unprojected(corpus: list) -> HierarchicalDialectClassifier:
    """AlwaysAccept, no projection: every sample answered at the finest level."""
    return _fit(corpus, policy=AlwaysAccept(), mask_to_coarse=False)


@pytest.fixture(scope="module")
def fitted_backoff(corpus: list) -> HierarchicalDialectClassifier:
    """An unreachable threshold + masking off: every sample backs off to family."""
    return _fit(corpus, policy=ConfidenceThreshold(0.999), mask_to_coarse=False)


def _prediction(
    pairs: list[tuple[str, float]],
    *,
    level: LabelLevel = LabelLevel.DIALECT,
    abstained: bool = False,
) -> Prediction:
    """Build a Prediction from ``(label, probability)`` pairs for policy tests."""
    probabilities = tuple(ClassProbability(label=label, probability=p) for label, p in pairs)
    return Prediction(
        label=pairs[0][0], level=level, probabilities=probabilities, abstained=abstained
    )


class TestBackoffLevels:
    def test_always_accept_stays_at_the_finest_level(
        self, fitted_always_unprojected: HierarchicalDialectClassifier, corpus: list
    ) -> None:
        predictions = fitted_always_unprojected.predict_samples(corpus)
        assert len(predictions) == len(corpus)
        assert all(p.level is LabelLevel.DIALECT for p in predictions)

    def test_a_standard_family_prediction_cannot_answer_at_dialect_level(
        self,
        fitted_always: HierarchicalDialectClassifier,
        fitted_backoff: HierarchicalDialectClassifier,
        corpus: list,
    ) -> None:
        """`standard` has no dialects, so the fine level must back off, not guess.

        Before this was fixed, projecting onto `standard` zeroed every dialect and
        the classifier quietly returned the *unprojected* dialect prediction --
        answering "which dialect?" for a sample it had just called non-dialectal.
        """
        dialect_predictions = fitted_always.predict_samples(corpus)
        family_predictions = fitted_backoff.predict_samples(corpus)

        standard = [
            prediction
            for prediction, family in zip(dialect_predictions, family_predictions, strict=True)
            if family.label == DialectFamily.STANDARD.value
        ]
        assert standard, "expected some samples predicted as standard Polish"
        assert all(p.level is LabelLevel.FAMILY for p in standard)
        assert all(p.label == DialectFamily.STANDARD.value for p in standard)

    def test_impossible_confidence_backs_off_to_family(
        self, fitted_backoff: HierarchicalDialectClassifier, corpus: list
    ) -> None:
        predictions = fitted_backoff.predict_samples(corpus)
        assert predictions
        assert all(p.level is LabelLevel.FAMILY for p in predictions)
        # Each backed-off label is a genuine DialectFamily value.
        for prediction in predictions:
            assert prediction.label in FAMILY_VALUES
            assert DialectFamily(prediction.label).value == prediction.label

    def test_backoff_walk_returns_one_prediction_per_sample(
        self, fitted_backoff: HierarchicalDialectClassifier, corpus: list
    ) -> None:
        assert len(fitted_backoff.predict_samples(corpus)) == len(corpus)

    def test_empty_input_yields_empty_output(
        self, fitted_always: HierarchicalDialectClassifier
    ) -> None:
        assert fitted_always.predict_samples([]) == []


class TestMasking:
    def test_accepted_fine_prediction_is_family_consistent(
        self,
        fitted_always: HierarchicalDialectClassifier,
        fitted_backoff: HierarchicalDialectClassifier,
        corpus: list,
    ) -> None:
        # Both fixtures train an identical FAMILY classifier (same seed/model),
        # so the family fitted_backoff reports is the family fitted_always masks
        # to. Where that family is a real (non-standard) group, the accepted
        # dialect must belong to it.
        dialect_predictions = fitted_always.predict_samples(corpus)
        family_predictions = fitted_backoff.predict_samples(corpus)

        checked = 0
        for dialect, family in zip(dialect_predictions, family_predictions, strict=True):
            if family.label == DialectFamily.STANDARD.value:
                continue  # standard has no dialects -> masking legitimately no-ops
            checked += 1
            assert dialect.level is LabelLevel.DIALECT
            resolved = family_for(dialect.label)
            assert resolved is not None
            assert resolved.value == family.label
        assert checked > 0, "expected some samples predicted to a real family"

    def test_masking_zeros_inconsistent_dialects(
        self, fitted_always: HierarchicalDialectClassifier, corpus: list
    ) -> None:
        # Projecting onto a family retains only that family's dialects; the rest
        # drop to exactly zero.
        predictions = fitted_always.predict_samples(corpus)
        assert any(any(cp.probability == 0.0 for cp in p.probabilities) for p in predictions)

    def test_masking_off_matches_a_plain_dialect_classifier(self, corpus: list) -> None:
        hierarchical = _fit(corpus, policy=AlwaysAccept(), mask_to_coarse=False)
        plain = DialectClassifier(
            model="logistic_regression",
            features=FEATURES,
            target=LabelLevel.DIALECT,
            seed=42,
        ).fit(corpus)

        for masked_off, plain_pred in zip(
            hierarchical.predict_samples(corpus), plain.predict_samples(corpus), strict=True
        ):
            assert masked_off.level is LabelLevel.DIALECT
            assert masked_off.label == plain_pred.label
            assert masked_off.as_dict() == pytest.approx(plain_pred.as_dict())

    def test_unprojected_probabilities_sum_to_one_and_are_never_all_zero(
        self, fitted_always_unprojected: HierarchicalDialectClassifier, corpus: list
    ) -> None:
        for prediction in fitted_always_unprojected.predict_samples(corpus):
            probs = [cp.probability for cp in prediction.probabilities]
            assert probs, "distribution is never empty"
            assert sum(probs) == pytest.approx(1.0)
            assert max(probs) > 0.0

    def test_a_projected_row_is_a_joint_bounded_by_its_family(
        self,
        fitted_always: HierarchicalDialectClassifier,
        fitted_backoff: HierarchicalDialectClassifier,
        corpus: list,
    ) -> None:
        """A projected dialect row sums to P(family) -- it is never renormalised to 1.

        Renormalising is the bug this pins: a single-dialect family (Kashubian ->
        Kashubia) would collapse to a certainty of 1.000 regardless of how unsure
        the family classifier was, defeating every confidence-based backoff policy.
        """
        fine_predictions = fitted_always.predict_samples(corpus)
        family_predictions = fitted_backoff.predict_samples(corpus)

        checked = 0
        for fine, family in zip(fine_predictions, family_predictions, strict=True):
            probs = [cp.probability for cp in fine.probabilities]
            assert probs and max(probs) > 0.0
            if fine.level is not LabelLevel.DIALECT:
                assert sum(probs) == pytest.approx(1.0)  # a backed-off family row
                continue
            checked += 1
            assert sum(probs) == pytest.approx(family.confidence)
            # The chain rule forbids a child from out-confidencing its parent.
            assert fine.confidence <= family.confidence + 1e-9
        assert checked > 0, "expected some samples answered at dialect level"

    def test_masking_skipped_for_non_family_dialect_pairs(self, corpus: list) -> None:
        # levels ordered DIALECT (coarse) -> REGION (fine): the coarser neighbour
        # is not FAMILY, so masking is inert and cannot raise.
        classifier = HierarchicalDialectClassifier(
            levels=(LabelLevel.DIALECT, LabelLevel.REGION),
            model="logistic_regression",
            features=FEATURES,
            policy=AlwaysAccept(),
            mask_to_coarse=True,
        ).fit(corpus)
        predictions = classifier.predict_samples(corpus)
        assert all(p.level is LabelLevel.REGION for p in predictions)


class TestProtocolAndSubstitutability:
    def test_is_a_sample_predictor(self, fitted_always: HierarchicalDialectClassifier) -> None:
        assert isinstance(fitted_always, SamplePredictor)

    def test_is_not_a_dialect_classifier_subclass(self) -> None:
        # Pinned so a future refactor cannot silently reintroduce inheritance from
        # DialectClassifier, whose fixed-level postcondition a backoff model cannot honour.
        assert not issubclass(HierarchicalDialectClassifier, DialectClassifier)


class TestDeterminismAndPersistence:
    def test_identical_seed_reproduces_predictions(self, corpus: list) -> None:
        first = _fit(corpus, policy=ConfidenceThreshold(0.6), seed=42)
        second = _fit(corpus, policy=ConfidenceThreshold(0.6), seed=42)

        first_preds = first.predict_samples(corpus)
        second_preds = second.predict_samples(corpus)
        assert [p.level for p in first_preds] == [p.level for p in second_preds]
        assert [p.label for p in first_preds] == [p.label for p in second_preds]
        for a, b in zip(first_preds, second_preds, strict=True):
            assert a.as_dict() == pytest.approx(b.as_dict())

    def test_save_load_round_trip_reproduces_predictions(self, corpus, tmp_path) -> None:
        classifier = _fit(
            corpus, policy=AllOf((ConfidenceThreshold(0.6), NotAbstained())), mask_to_coarse=True
        )
        original = classifier.predict_samples(corpus)
        # A meaningful round trip: the policy actually mixes both levels.
        levels_seen = {p.level for p in original}
        assert levels_seen == {LabelLevel.DIALECT, LabelLevel.FAMILY}

        classifier.save(tmp_path / "hier")
        restored = HierarchicalDialectClassifier.load(tmp_path / "hier")

        assert restored.levels == classifier.levels
        assert restored.mask_to_coarse is True
        assert isinstance(restored.policy, AllOf)
        for before, after in zip(original, restored.predict_samples(corpus), strict=True):
            assert before.label == after.label
            assert before.level is after.level
            assert before.as_dict() == pytest.approx(after.as_dict())

    def test_load_rejects_a_foreign_kind(self, tmp_path) -> None:
        root = tmp_path / "foreign"
        root.mkdir()
        (root / "hierarchical.json").write_text(
            '{"kind": "SomethingElse", "config": {}}', encoding="utf-8"
        )
        with pytest.raises(DataError, match="not saved by HierarchicalDialectClassifier"):
            HierarchicalDialectClassifier.load(root)

    def test_load_missing_sidecar_raises(self, corpus, tmp_path) -> None:
        # A plain classifier artifact lacks the hierarchical sidecar entirely.
        DialectClassifier(model="naive_bayes", features=["char_tfidf"]).fit(corpus).save(
            tmp_path / "plain"
        )
        with pytest.raises(DataError, match="missing hierarchical"):
            HierarchicalDialectClassifier.load(tmp_path / "plain")

    def test_load_absent_directory_raises(self, tmp_path) -> None:
        with pytest.raises(DataError, match="missing hierarchical"):
            HierarchicalDialectClassifier.load(tmp_path / "nowhere")

    def test_unserialisable_policy_is_reported_on_save(self, corpus, tmp_path) -> None:
        class _CustomPolicy:
            def accepts(self, prediction: Prediction) -> bool:
                return True

        classifier = _fit(corpus, policy=_CustomPolicy())
        with pytest.raises(ConfigurationError, match="not serialisable"):
            classifier.save(tmp_path / "custom")


class TestConfigValidation:
    def test_fewer_than_two_levels_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="at least two"):
            HierarchicalDialectClassifier(levels=(LabelLevel.DIALECT,), model="logistic_regression")

    def test_duplicate_levels_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="unique"):
            HierarchicalDialectClassifier(
                levels=(LabelLevel.FAMILY, LabelLevel.FAMILY), model="logistic_regression"
            )

    def test_unfitted_predict_raises(self, corpus: list) -> None:
        classifier = HierarchicalDialectClassifier(levels=LEVELS, model="logistic_regression")
        with pytest.raises(ConfigurationError, match="not fitted"):
            classifier.predict_samples(corpus[:1])


class TestPolicies:
    def test_confidence_threshold(self) -> None:
        prediction = _prediction([("a", 0.6), ("b", 0.4)])
        assert ConfidenceThreshold(0.55).accepts(prediction)
        assert not ConfidenceThreshold(0.7).accepts(prediction)

    def test_margin_threshold_uses_top1_minus_top2(self) -> None:
        prediction = _prediction([("a", 0.6), ("b", 0.5), ("c", 0.0)])
        assert MarginThreshold(0.05).accepts(prediction)  # margin is 0.1
        assert not MarginThreshold(0.2).accepts(prediction)

    def test_margin_threshold_accepts_single_class(self) -> None:
        assert MarginThreshold(0.9).accepts(_prediction([("a", 1.0)]))

    def test_not_abstained(self) -> None:
        assert NotAbstained().accepts(_prediction([("a", 0.3)], abstained=False))
        assert not NotAbstained().accepts(_prediction([("a", 0.3)], abstained=True))

    def test_always_accept_is_a_null_object(self) -> None:
        # Accepts even an abstained, low-confidence prediction.
        assert AlwaysAccept().accepts(_prediction([("a", 0.1)], abstained=True))

    def test_all_of_requires_every_child(self) -> None:
        prediction = _prediction([("a", 0.6), ("b", 0.4)], abstained=False)
        assert AllOf((ConfidenceThreshold(0.5), NotAbstained())).accepts(prediction)
        assert not AllOf((ConfidenceThreshold(0.7), NotAbstained())).accepts(prediction)

    def test_any_of_requires_one_child(self) -> None:
        prediction = _prediction([("a", 0.6), ("b", 0.4)], abstained=False)
        assert AnyOf((ConfidenceThreshold(0.9), NotAbstained())).accepts(prediction)
        assert not AnyOf((ConfidenceThreshold(0.9), MarginThreshold(0.9))).accepts(prediction)

    def test_default_policy_is_always_accept(self) -> None:
        classifier = HierarchicalDialectClassifier(levels=LEVELS, model="logistic_regression")
        assert isinstance(classifier.policy, AlwaysAccept)

    @pytest.mark.parametrize(
        "policy",
        [
            ConfidenceThreshold(0.7),
            MarginThreshold(0.1),
            NotAbstained(),
            AlwaysAccept(),
            AllOf((ConfidenceThreshold(0.5), NotAbstained())),
            AnyOf((ConfidenceThreshold(0.9), MarginThreshold(0.05))),
        ],
    )
    def test_policy_spec_round_trip(self, policy: object) -> None:
        rebuilt = policy_from_spec(policy.to_spec())  # type: ignore[attr-defined]
        assert rebuilt == policy

    def test_policy_from_spec_rejects_unknown_kind(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown backoff policy"):
            policy_from_spec(PolicySpec(kind="does_not_exist"))
