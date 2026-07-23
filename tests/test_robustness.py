"""Tests for robustness under linguistic perturbation."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from conftest import make_manifest_experiment_config, make_samples, write_manifest_corpus
from tulip.data.augment import augment_samples
from tulip.features.text._tokenize import word_tokens
from tulip.features.text.dialect_intensity import DialectIntensityExtractor
from tulip.features.text.phonological_rules import (
    apply_rules,
    load_phonological_rules,
    normalize_to_standard,
)
from tulip.robustness import (
    PERTURBATIONS,
    AugmentSpec,
    PerturbationConfig,
    RobustnessCell,
    RobustnessCurve,
    RobustnessReport,
    perturb_samples,
    run_robustness,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tulip.config.schemas import ExperimentConfig

# Tokens carrying soft-labial environments (pi, wi, mi, bi), so the forward dial
# produces detectable dialectal reflexes and the intensity extractor sees them.
_SOFT_LABIAL_TEXT = " ".join(["piwo wiosna miasto bialy pies"] * 20)
_ALL_NAMES = ("asr_noise", "dialect_intensity_dial", "standardize", "typo_noise")


def _overall_intensity(text: str) -> float:
    extractor = DialectIntensityExtractor().fit([""])
    return float(extractor.transform([text])[0][0])


def _total_fired_rate(text: str) -> float:
    rules = load_phonological_rules()
    tokens = word_tokens(text, lowercase=True)
    return sum(rule.fired_rate(tokens) for rule in rules)


# ------------------------------------------------------------------ registry


def test_all_builtins_registered_and_creatable() -> None:
    assert set(PERTURBATIONS.names()) == set(_ALL_NAMES)
    for name in _ALL_NAMES:
        perturbation = PERTURBATIONS.create(name)
        assert hasattr(perturbation, "perturb")


# --------------------------------------------------------------- perturbations


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_level_zero_is_identity(name: str) -> None:
    perturbation = PERTURBATIONS.create(name)
    text = "Zażółć gęślą jaźń, piwo i wiosna."
    assert perturbation.perturb(text, level=0.0, rng=np.random.default_rng(0)) == text


def test_dial_level_one_equals_apply_rules() -> None:
    dial = PERTURBATIONS.create("dialect_intensity_dial")
    text = "piwo wiosna miasto bialy pies"
    result = dial.perturb(text, level=1.0, rng=np.random.default_rng(0))
    assert result == apply_rules(text)


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_perturbation_is_deterministic(name: str) -> None:
    perturbation = PERTURBATIONS.create(name)
    text = "Zażółć gęślą jaźń, piwo w mieście."
    first = perturbation.perturb(text, level=0.5, rng=np.random.default_rng(7))
    second = perturbation.perturb(text, level=0.5, rng=np.random.default_rng(7))
    assert first == second


def test_dial_raises_measured_intensity_monotonically() -> None:
    # Same seed per level nests the rewritten token sets, so intensity can only
    # rise as the level rises. This is the linguistic-validity guard.
    dial = PERTURBATIONS.create("dialect_intensity_dial")
    intensities = [
        _overall_intensity(
            dial.perturb(_SOFT_LABIAL_TEXT, level=level, rng=np.random.default_rng(0))
        )
        for level in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert intensities == sorted(intensities)
    assert intensities[-1] > intensities[0]


def test_standardize_lowers_intensity() -> None:
    dialectal = apply_rules(_SOFT_LABIAL_TEXT)
    standardize = PERTURBATIONS.create("standardize")
    before = _overall_intensity(dialectal)
    after = _overall_intensity(
        standardize.perturb(dialectal, level=1.0, rng=np.random.default_rng(0))
    )
    assert after < before


def test_round_trip_does_not_raise_fired_rate() -> None:
    forward = apply_rules(_SOFT_LABIAL_TEXT)
    back = normalize_to_standard(forward)
    assert _total_fired_rate(back) <= _total_fired_rate(forward)


# ----------------------------------------------------------- perturb_samples


def test_perturb_samples_is_deterministic() -> None:
    samples = make_samples(repeats=2)
    dial = PERTURBATIONS.create("dialect_intensity_dial")
    first = perturb_samples(samples, dial, 0.5, seed=(0, 1))
    second = perturb_samples(samples, dial, 0.5, seed=(0, 1))
    assert [s.text for s in first] == [s.text for s in second]
    assert [s.id for s in first] == [s.id for s in second]


def test_perturb_samples_keeps_ids_and_labels() -> None:
    samples = make_samples(repeats=2)
    perturbed = perturb_samples(samples, PERTURBATIONS.create("asr_noise"), 1.0, seed=0)
    assert [s.id for s in perturbed] == [s.id for s in samples]
    assert [s.labels for s in perturbed] == [s.labels for s in samples]


# ----------------------------------------------------------------- augment


def test_augment_grows_and_is_deterministic() -> None:
    samples = make_samples(repeats=2)
    spec = AugmentSpec(
        perturbations=(PerturbationConfig(name="dialect_intensity_dial"),), multiplier=2, seed=3
    )
    first = augment_samples(samples, spec)
    second = augment_samples(samples, spec)
    assert len(first) == len(samples) + 2 * len(samples)
    assert [s.id for s in first] == [s.id for s in second]
    assert [s.text for s in first] == [s.text for s in second]
    assert first[: len(samples)] == list(samples)  # originals preserved first


def test_negative_seed_is_rejected_at_config_construction() -> None:
    from pydantic import ValidationError

    # Both seeds feed numpy's default_rng, which rejects a negative value; the
    # constraint lives on the config so a bad seed is caught when the config is
    # built rather than crashing mid-run inside the rng.
    with pytest.raises(ValidationError):
        PerturbationConfig(name="typo_noise", seed=-1)
    with pytest.raises(ValidationError):
        AugmentSpec(perturbations=(PerturbationConfig(name="typo_noise"),), seed=-1)


# ------------------------------------------------------------------ report


def _tiny_report() -> RobustnessReport:
    baseline = RobustnessCell(perturbation="clean", level=0.0, n=10, accuracy=0.9, f1_macro=0.88)
    curve = RobustnessCurve(
        perturbation="asr_noise",
        clean_f1=0.88,
        cells=(
            RobustnessCell(perturbation="asr_noise", level=0.0, n=10, accuracy=0.9, f1_macro=0.88),
            RobustnessCell(perturbation="asr_noise", level=1.0, n=10, accuracy=0.5, f1_macro=0.5),
        ),
    )
    return RobustnessReport(
        model="logistic_regression", target="dialect", baseline=baseline, curves=(curve,)
    )


def test_curve_derived_fields() -> None:
    curve = _tiny_report().curves[0]
    assert curve.degradation_slope == pytest.approx((0.5 - 0.88) / 1.0)
    assert curve.breakdown_level == 1.0  # 0.5 < 0.8 * 0.88


def test_report_markdown_and_save_is_byte_stable(tmp_path: Path) -> None:
    report = _tiny_report()
    markdown = report.to_markdown()
    assert "# Robustness - logistic_regression (dialect)" in markdown
    assert "asr_noise" in markdown
    report.save(tmp_path / "a.json")
    report.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_report_import_is_light() -> None:
    code = (
        "import sys, tulip.robustness.report as _;"
        "heavy=[m for m in ('sklearn','torch') if m in sys.modules];"
        "raise SystemExit(1 if heavy else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)  # noqa: S603  (trusted, fixed input)
    assert result.returncode == 0


# ------------------------------------------------------------------ sweep


@pytest.fixture
def sweep_config(tmp_path: Path) -> ExperimentConfig:
    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=2)
    return make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="robust")


def test_run_robustness_deterministic_and_degrades(
    sweep_config: ExperimentConfig, tmp_path: Path
) -> None:
    samples = make_samples(repeats=6)
    specs = [PerturbationConfig(name="asr_noise", levels=(0.0, 0.5, 1.0), seed=0)]
    first = run_robustness(sweep_config, perturbations=specs, samples=samples)
    second = run_robustness(sweep_config, perturbations=specs, samples=samples)

    first.save(tmp_path / "first.json")
    second.save(tmp_path / "second.json")
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    assert first.to_markdown() == second.to_markdown()

    curve = first.curves[0]
    # The level-0 cell is identity, so it equals the clean baseline exactly.
    assert curve.cells[0].f1_macro == first.baseline.f1_macro
    # Clean is at least as good as the worst perturbed cell, within noise.
    worst = min(cell.f1_macro for cell in curve.cells)
    assert first.baseline.f1_macro >= worst - 0.1
