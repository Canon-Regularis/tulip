"""Tests for tulip.evaluation.calibration and its wiring into compute_metrics.

The synthetic generators here are legitimate ground truth: a perfectly
calibrated stream is built by drawing each label with probability equal to the
model's stated confidence, so ``accuracy == confidence`` holds in expectation
and ECE must vanish. Sharpening those same probabilities (``p ** 3``) leaves the
labels -- and thus the accuracy -- untouched while inflating confidence, so the
calibration error can only grow. We assert those *directions*, never magic
numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.evaluation.calibration import (
    CalibrationBin,
    CalibrationReport,
    compute_calibration,
    reliability_curve,
)
from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.report import EvaluationReport

if TYPE_CHECKING:
    from pathlib import Path


class TestProbabilityValueValidation:
    """Bad probability VALUES must fail like bad shapes do: one clean error."""

    @pytest.mark.parametrize(
        ("tag", "proba"),
        [
            ("nan", [[float("nan"), 0.5], [0.4, 0.6]]),
            ("inf", [[float("inf"), 0.0], [0.4, 0.6]]),
        ],
    )
    def test_non_finite_probabilities_raise_configuration_error(
        self, tag: str, proba: list[list[float]]
    ) -> None:
        # A NaN row makes argmax select the NaN slot; the bin's mean confidence
        # then tripped CalibrationBin's Field(ge=0, le=1) and surfaced as a
        # cryptic pydantic error about an internal value object.
        with pytest.raises(ConfigurationError, match="non-finite"):
            compute_calibration(["0", "1"], proba, ["0", "1"], n_bins=3)

    def test_out_of_range_probabilities_raise_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match=r"\[0, 1\]"):
            compute_calibration(["0", "1"], [[1.4, 0.1], [0.4, 0.6]], ["0", "1"], n_bins=3)

    def test_a_value_at_the_boundary_is_accepted(self) -> None:
        report = compute_calibration(["0", "1"], [[1.0, 0.0], [0.0, 1.0]], ["0", "1"], n_bins=3)
        assert report.ece == pytest.approx(0.0)


# Binary micro-case shared with the metrics tests, so calibration wiring is
# exercised on inputs whose standard metrics are already pinned elsewhere.
Y_TRUE = ["a", "a", "b", "b"]
Y_PRED = ["a", "b", "b", "b"]
Y_PROBA = [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]

_LABELS = ["a", "b", "c"]


def _calibrated_stream(n: int, seed: int = 7) -> tuple[list[str], np.ndarray]:
    """Build a perfectly calibrated 3-class stream.

    The predicted class always sits in column 0 with probability ``p`` (the
    confidence); the label equals that class with probability exactly ``p``, so
    accuracy tracks confidence and ECE is zero in expectation.
    """
    rng = np.random.default_rng(seed)
    confidence = rng.uniform(1.0 / 3.0, 1.0, size=n)
    proba = np.empty((n, 3))
    proba[:, 0] = confidence
    proba[:, 1] = (1.0 - confidence) / 2.0
    proba[:, 2] = (1.0 - confidence) / 2.0
    hit = rng.random(n) < confidence
    other = rng.integers(1, 3, size=n)  # 1 or 2 when the model is wrong
    true_index = np.where(hit, 0, other)
    y_true = [_LABELS[i] for i in true_index]
    return y_true, proba


def _random_stream(n: int, k: int, seed: int = 0) -> tuple[list[str], np.ndarray, list[str]]:
    """Arbitrary normalised probabilities with random labels (for invariants)."""
    rng = np.random.default_rng(seed)
    weights = rng.random((n, k))
    proba = weights / weights.sum(axis=1, keepdims=True)
    labels = [f"c{i}" for i in range(k)]
    y_true = [labels[i] for i in rng.integers(0, k, size=n)]
    return y_true, proba, labels


def test_perfect_calibration_has_near_zero_ece() -> None:
    y_true, proba = _calibrated_stream(50_000)
    report = compute_calibration(y_true, proba, _LABELS, n_bins=15)
    assert report.ece < 0.02
    assert report.mce < 0.1


def test_overconfidence_inflates_ece() -> None:
    y_true, proba = _calibrated_stream(50_000)
    calibrated = compute_calibration(y_true, proba, _LABELS, n_bins=15)
    # Sharpen the (argmax-preserving) probabilities: same labels, same accuracy,
    # higher confidence -> the model becomes overconfident.
    sharp = proba**3
    sharp /= sharp.sum(axis=1, keepdims=True)
    overconfident = compute_calibration(y_true, sharp, _LABELS, n_bins=15)
    assert overconfident.ece > calibrated.ece
    assert overconfident.ece > 0.05  # materially larger, not just noise


def test_confident_and_correct_one_hot_is_perfectly_calibrated() -> None:
    y_true = ["a", "b", "c", "a", "b"]
    index = {label: i for i, label in enumerate(_LABELS)}
    proba = np.zeros((len(y_true), 3))
    for row, label in enumerate(y_true):
        proba[row, index[label]] = 1.0
    report = compute_calibration(y_true, proba, _LABELS, n_bins=15)
    assert report.ece == pytest.approx(0.0)
    # Every sample has confidence 1.0, so only the top bin is populated; the
    # empty bins are dropped without disturbing ECE.
    assert len(report.bins) == 1
    assert report.bins[0].count == len(y_true)


def test_confident_and_wrong_one_hot_has_maximal_ece() -> None:
    y_true = ["a", "b", "c", "a", "b"]
    index = {label: i for i, label in enumerate(_LABELS)}
    proba = np.zeros((len(y_true), 3))
    for row, label in enumerate(y_true):
        proba[row, (index[label] + 1) % 3] = 1.0  # certain, and always wrong
    report = compute_calibration(y_true, proba, _LABELS, n_bins=15)
    assert report.ece == pytest.approx(1.0)
    assert report.mce == pytest.approx(1.0)


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
@pytest.mark.parametrize("n_bins", [1, 5, 20])
def test_calibration_invariants_over_random_inputs(strategy: str, n_bins: int) -> None:
    y_true, proba, labels = _random_stream(600, 4, seed=n_bins)
    report = compute_calibration(y_true, proba, labels, n_bins=n_bins, strategy=strategy)
    assert 0.0 <= report.ece <= 1.0
    assert report.ece <= report.mce + 1e-12
    assert 0.0 <= report.brier <= 2.0
    assert sum(bin_.count for bin_ in report.bins) == report.n_samples == 600


def test_brier_matches_manual_definition() -> None:
    # Two samples, three classes; hand-compute mean_i sum_k (p_ik - 1[y==k])^2.
    y_true = ["a", "b"]
    proba = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
    # sample 0 (true a): (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
    # sample 1 (true b): 0.3^2 + (0.4-1)^2 + 0.3^2 = 0.09 + 0.36 + 0.09 = 0.54
    report = compute_calibration(y_true, proba, _LABELS, n_bins=5)
    assert report.brier == pytest.approx((0.14 + 0.54) / 2)


def test_quantile_strategy_balances_bin_counts() -> None:
    y_true, proba, labels = _random_stream(1_000, 4, seed=3)
    quantile = compute_calibration(y_true, proba, labels, n_bins=10, strategy="quantile")
    uniform = compute_calibration(y_true, proba, labels, n_bins=10, strategy="uniform")

    q_counts = [bin_.count for bin_ in quantile.bins]
    assert len(q_counts) == 10  # continuous confidences => no collapsed edges
    # Equal-count bins: the spread stays within a small fraction of the mean...
    assert max(q_counts) - min(q_counts) <= 0.1 * (1_000 / 10)
    # ...and is far tighter than equal-width bins on the same (peaked) data.
    u_counts = [bin_.count for bin_ in uniform.bins]
    assert max(q_counts) - min(q_counts) < max(u_counts) - min(u_counts)


def test_reliability_curve_aligns_with_bins() -> None:
    y_true, proba, labels = _random_stream(400, 3, seed=1)
    report = compute_calibration(y_true, proba, labels, n_bins=8)
    confidence, accuracy, count = reliability_curve(report)
    assert confidence.shape == accuracy.shape == count.shape == (len(report.bins),)
    assert confidence.tolist() == [bin_.confidence for bin_ in report.bins]
    assert count.sum() == report.n_samples


def test_to_markdown_reports_headline_metrics() -> None:
    y_true, proba, labels = _random_stream(200, 3, seed=2)
    markdown = compute_calibration(y_true, proba, labels, n_bins=6).to_markdown()
    assert "# Calibration report" in markdown
    assert "ECE" in markdown
    assert "Per-bin reliability" in markdown


def test_unknown_strategy_raises() -> None:
    y_true, proba, labels = _random_stream(20, 3)
    with pytest.raises(ConfigurationError, match="unknown calibration strategy"):
        compute_calibration(y_true, proba, labels, strategy="sigmoid")


def test_non_positive_bins_raise() -> None:
    y_true, proba, labels = _random_stream(20, 3)
    with pytest.raises(ConfigurationError, match="n_bins"):
        compute_calibration(y_true, proba, labels, n_bins=0)


def test_empty_input_raises() -> None:
    with pytest.raises(ConfigurationError, match="zero samples"):
        compute_calibration([], np.empty((0, 3)), _LABELS)


def test_duplicate_labels_raise() -> None:
    y_true, proba, _ = _random_stream(20, 3)
    with pytest.raises(ConfigurationError, match="duplicates"):
        compute_calibration(y_true, proba, ["a", "b", "a"])


def test_row_count_mismatch_raises() -> None:
    y_true, proba, labels = _random_stream(20, 3)
    with pytest.raises(ConfigurationError, match="rows"):
        compute_calibration(y_true, proba[:-1], labels)


def test_column_count_mismatch_raises() -> None:
    y_true, proba, _ = _random_stream(20, 3)
    with pytest.raises(ConfigurationError, match="columns"):
        compute_calibration(y_true, proba, ["a", "b"])


def test_true_label_absent_from_labels_raises() -> None:
    proba = np.full((3, 3), 1 / 3)
    with pytest.raises(ConfigurationError, match="missing from"):
        compute_calibration(["a", "b", "z"], proba, _LABELS)


def test_non_numeric_proba_raises() -> None:
    with pytest.raises(ConfigurationError, match="numeric"):
        compute_calibration(["a", "b"], "nonsense", _LABELS)


# --- compute_metrics wiring: the field defaults off and must not drift ------


def test_compute_metrics_leaves_calibration_none_by_default() -> None:
    # Guard against artifact drift: the standard flow never touches calibration.
    assert compute_metrics(Y_TRUE, Y_PRED, y_proba=Y_PROBA).calibration is None
    assert compute_metrics(Y_TRUE, Y_PRED).calibration is None


def test_compute_metrics_populates_calibration_and_round_trips(tmp_path: Path) -> None:
    report = compute_metrics(Y_TRUE, Y_PRED, y_proba=Y_PROBA, calibration_bins=10)
    assert report.calibration is not None
    assert report.calibration.n_bins == 10
    assert report.calibration.n_samples == len(Y_TRUE)

    path = tmp_path / "report.json"
    report.save(path)
    loaded = EvaluationReport.load(path)
    assert loaded == report
    assert loaded.calibration == report.calibration


def test_compute_metrics_calibration_skipped_when_proba_unusable() -> None:
    # calibration_bins requested but no probabilities: field stays None, no raise.
    assert compute_metrics(Y_TRUE, Y_PRED, calibration_bins=10).calibration is None
    # Wrong column count is unusable for calibration but must not fail metrics.
    bad = np.full((len(Y_TRUE), 3), 1 / 3)  # 3 columns, 2 labels
    assert compute_metrics(Y_TRUE, Y_PRED, y_proba=bad, calibration_bins=10).calibration is None


# --- reliability diagram (viz extra) ----------------------------------------


def test_reliability_diagram_matplotlib_has_two_panels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    y_true, proba, labels = _random_stream(300, 3, seed=5)
    report = compute_calibration(y_true, proba, labels, n_bins=10)

    from tulip.viz.charts import reliability_diagram

    fig = reliability_diagram(report, title="Calibration")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2  # reliability panel + count histogram


def test_reliability_diagram_rejects_unknown_backend_and_empty_report() -> None:
    pytest.importorskip("matplotlib")
    from tulip.viz.charts import reliability_diagram

    y_true, proba, labels = _random_stream(100, 3, seed=6)
    report = compute_calibration(y_true, proba, labels, n_bins=5)
    with pytest.raises(ConfigurationError):
        reliability_diagram(report, backend="ascii")

    empty = CalibrationReport(
        n_bins=5, n_samples=1, ece=0.0, mce=0.0, brier=0.0, bins=(), strategy="uniform"
    )
    with pytest.raises(DataError):
        reliability_diagram(empty)


def test_calibration_bin_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="less than or equal to 1"):
        CalibrationBin(lower=0.0, upper=1.5, confidence=0.5, accuracy=0.5, count=1)
