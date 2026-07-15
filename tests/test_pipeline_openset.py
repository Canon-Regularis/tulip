"""Tests for open-set novelty detection over a conformal classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from conftest import make_samples
from tulip.pipeline import ConformalClassifier, DialectClassifier, OpenSetClassifier

if TYPE_CHECKING:
    from tulip.core.types import Sample


def _speaker_index(sample: Sample) -> int:
    assert sample.speaker_id is not None
    return int(sample.speaker_id.rsplit("spk", 1)[1])


def _openset_splits() -> tuple[list[Sample], list[Sample], list[Sample], list[Sample]]:
    """Train and calibrate on two dialects; the third is the unseen (novel) one."""
    labelled = [s for s in make_samples(repeats=6) if s.labels.dialect]
    dialects = sorted({s.labels.dialect for s in labelled if s.labels.dialect})
    novel_dialect = dialects[-1]
    known = [s for s in labelled if s.labels.dialect != novel_dialect]
    novel = [s for s in labelled if s.labels.dialect == novel_dialect]
    train = [s for s in known if _speaker_index(s) <= 3]
    calibration = [s for s in known if _speaker_index(s) == 4]
    test_known = [s for s in known if _speaker_index(s) == 5]
    return train, calibration, test_known, novel


def _fit_openset(
    train: list[Sample], calibration: list[Sample], *, alpha: float = 0.1
) -> OpenSetClassifier:
    base = DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=0).fit(
        train
    )
    conformal = ConformalClassifier(base, alpha=alpha)
    conformal.fit_conformal(calibration)
    return OpenSetClassifier(conformal)


def test_coverage_holds_on_calibration() -> None:
    # On the calibration set the conformal set covers the truth at 1 - alpha, so
    # known coverage must clear that bar and nothing there is novel.
    train, calibration, _, _ = _openset_splits()
    report = _fit_openset(train, calibration, alpha=0.1).evaluate(calibration)
    assert report.n_novel == 0
    assert report.novelty_auroc is None  # one class only, AUROC undefined
    assert report.known_coverage >= 0.85


def test_scores_and_reports_novel() -> None:
    train, calibration, test_known, novel = _openset_splits()
    report = _fit_openset(train, calibration, alpha=0.1).evaluate(test_known + novel)
    assert report.n_known > 0
    assert report.n_novel > 0
    assert report.novelty_auroc is not None
    assert 0.0 <= report.novelty_auroc <= 1.0
    assert report.detection_rate is not None
    assert "Open-set" in report.to_markdown()


def test_prediction_fields_are_consistent() -> None:
    train, calibration, test_known, _ = _openset_splits()
    openset = _fit_openset(train, calibration)
    predictions = openset.predict_openset([s.text or "" for s in test_known])
    assert len(predictions) == len(test_known)
    for prediction in predictions:
        assert 0.0 <= prediction.novelty_score <= 1.0
        # in-distribution means a non-empty conformal set, novel means empty.
        assert prediction.in_distribution == bool(prediction.prediction_set)
        assert prediction.top_label in openset.classes_
