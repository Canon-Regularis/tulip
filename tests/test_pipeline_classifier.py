"""End-to-end tests for the DialectClassifier facade (no optional deps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import make_samples
from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import Prediction, Sample, TaskType
from tulip.labels.taxonomy import LabelLevel
from tulip.pipeline import DialectClassifier

PODHALE_QUERY = "Hej, baca się pyto, kaj się owce pasą na holi."


@pytest.fixture(scope="module")
def fitted() -> DialectClassifier:
    classifier = DialectClassifier(
        model={"name": "logistic_regression"},
        features=[{"name": "char_tfidf"}, {"name": "word_tfidf"}],
        target=LabelLevel.DIALECT,
        seed=42,
    )
    return classifier.fit(make_samples(repeats=3))


class TestFitAndPredict:
    def test_predict_returns_full_sorted_distribution(self, fitted: DialectClassifier) -> None:
        prediction = fitted.predict(PODHALE_QUERY)
        assert isinstance(prediction, Prediction)
        assert prediction.level is LabelLevel.DIALECT
        probs = [cp.probability for cp in prediction.probabilities]
        assert probs == sorted(probs, reverse=True)
        assert np.isclose(sum(probs), 1.0)
        assert set(fitted.classes_) == {"podhale", "silesia", "kurpie"}

    def test_marked_query_ranks_its_dialect_first(self, fitted: DialectClassifier) -> None:
        prediction = fitted.predict(PODHALE_QUERY)
        assert prediction.label == "podhale"
        assert prediction.top_k(3)[0].label == "podhale"

    def test_predict_batch_matches_predict(self, fitted: DialectClassifier) -> None:
        queries = [PODHALE_QUERY, "Jo żech je z Katowic i godom po naszymu."]
        batch = fitted.predict_batch(queries)
        assert [p.label for p in batch] == [fitted.predict(q).label for q in queries]

    def test_standard_samples_without_dialect_label_are_skipped(self) -> None:
        # make_samples() labels standard sentences at family level only, so a
        # DIALECT-target classifier must silently train on the three dialects.
        classifier = DialectClassifier(
            model="naive_bayes", features=["char_tfidf"], target=LabelLevel.DIALECT
        )
        classifier.fit(make_samples())
        assert set(classifier.classes_) == {"podhale", "silesia", "kurpie"}

    def test_family_target_uses_family_labels(self) -> None:
        classifier = DialectClassifier(
            model="logistic_regression", features=["char_tfidf"], target=LabelLevel.FAMILY
        )
        classifier.fit(make_samples())
        # Families derived from dialects plus the explicit standard family.
        assert set(classifier.classes_) == {"lesser_polish", "silesian", "masovian", "standard"}

    def test_unfitted_predict_raises(self) -> None:
        classifier = DialectClassifier(model="naive_bayes", features=["char_tfidf"])
        with pytest.raises(ConfigurationError, match="not fitted"):
            classifier.predict("cokolwiek")

    def test_fit_with_nothing_trainable_raises(self) -> None:
        audio_only = [
            Sample(id="a", audio_path=Path("x.wav"), speaker_id="s1"),
        ]
        classifier = DialectClassifier(model="naive_bayes", features=["char_tfidf"])
        with pytest.raises(DataError, match="trainable"):
            classifier.fit(audio_only)

    def test_invalid_abstain_threshold_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="abstain_threshold"):
            DialectClassifier(model="naive_bayes", abstain_threshold=1.5)


class TestAbstention:
    def test_high_threshold_abstains_on_ambiguous_input(self) -> None:
        classifier = DialectClassifier(
            model="logistic_regression",
            features=["char_tfidf"],
            abstain_threshold=0.99,
            seed=42,
        )
        classifier.fit(make_samples())
        prediction = classifier.predict("To jest zupełnie neutralne zdanie o niczym.")
        assert prediction.abstained
        assert prediction.label is None
        assert prediction.probabilities  # distribution still reported

    def test_confident_prediction_is_not_abstained(self, fitted: DialectClassifier) -> None:
        prediction = fitted.predict(PODHALE_QUERY)
        assert not prediction.abstained


class TestExplain:
    def test_top_tfidf_surfaces_marker_evidence(self, fitted: DialectClassifier) -> None:
        explanation = fitted.explain(PODHALE_QUERY, method="top_tfidf")
        assert explanation.method == "top_tfidf"
        assert explanation.attributions
        strongest = explanation.top_attributions(5)
        assert any(a.weight > 0 for a in strongest)

    def test_nearest_examples_surfaces_training_sentence(self, fitted: DialectClassifier) -> None:
        explanation = fitted.explain(PODHALE_QUERY, method="nearest_examples", k=3)
        assert explanation.neighbors
        best = explanation.neighbors[0]
        assert best.label == "podhale"
        assert "baca" in (best.text or "")

    def test_unknown_method_lists_suggestions(self, fitted: DialectClassifier) -> None:
        from tulip.core.exceptions import UnknownComponentError

        with pytest.raises(UnknownComponentError):
            fitted.explain(PODHALE_QUERY, method="topp_tfidf")


class TestPersistence:
    def test_save_load_round_trip_preserves_predictions(
        self, fitted: DialectClassifier, tmp_path: Path
    ) -> None:
        queries = [PODHALE_QUERY, "U nos w boru psiwo warzą jesce po staremu."]
        fitted.save(tmp_path / "artifact")
        restored = DialectClassifier.load(tmp_path / "artifact")
        assert restored.classes_ == fitted.classes_
        assert restored.target is fitted.target
        assert restored.task is TaskType.TEXT
        for original, loaded in zip(
            fitted.predict_batch(queries), restored.predict_batch(queries), strict=True
        ):
            assert original.label == loaded.label
            assert original.as_dict() == pytest.approx(loaded.as_dict())

    def test_loaded_classifier_cannot_do_nearest_examples(
        self, fitted: DialectClassifier, tmp_path: Path
    ) -> None:
        fitted.save(tmp_path / "artifact")
        restored = DialectClassifier.load(tmp_path / "artifact")
        with pytest.raises(ConfigurationError, match="refitted"):
            restored.explain(PODHALE_QUERY, method="nearest_examples")

    def test_load_rejects_foreign_artifacts(self, tmp_path: Path) -> None:
        from sklearn.dummy import DummyClassifier

        from tulip.models.persistence import save_model

        save_model(DummyClassifier(), tmp_path / "foreign", metadata={"kind": "other"})
        with pytest.raises(DataError, match="DialectClassifier"):
            DialectClassifier.load(tmp_path / "foreign")
