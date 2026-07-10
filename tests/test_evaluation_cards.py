"""Tests for tulip.evaluation.cards (deterministic dataset and model cards).

Hermetic: cards are built from a tiny synthetic corpus and a tiny classifier
trained in-process. The load-bearing properties under test are that cards
render the key facts (class names, split sizes, metrics), are byte-stable for
identical inputs (they get committed to the repo), and degrade to ``"n/a"``
rather than raising when optional artifact fields are absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from conftest import make_samples, write_manifest_corpus
from tulip.config.schemas import ComponentConfig, DataConfig, SplitConfig
from tulip.core.types import DatasetInfo
from tulip.data.builder import DatasetBuilder
from tulip.data.splitting import speaker_disjoint_split
from tulip.evaluation.cards import dataset_card, dataset_card_from_splits, model_card
from tulip.evaluation.metrics import compute_metrics
from tulip.labels.taxonomy import LabelLevel, display_name
from tulip.models.persistence import load_model
from tulip.pipeline import DialectClassifier
from tulip.utils.io import read_json

if TYPE_CHECKING:
    from pathlib import Path

    from tulip.evaluation.report import EvaluationReport

_INFO = DatasetInfo(
    name="synthetic",
    description="A tiny synthetic dialect corpus.",
    url="https://example.org/synthetic",
    tier=4,
    tasks=("text",),
    label_levels=(LabelLevel.DIALECT,),
    license="CC0-1.0",
)


def _build_manifest(tmp_path: Path) -> dict:
    """Build a real dataset via DatasetBuilder and return its build manifest."""
    corpus = write_manifest_corpus(tmp_path / "corpus")
    config = DataConfig(
        datasets=[ComponentConfig(name="manifest", params={"root": str(corpus)})],
        root=corpus.parent,
        deduplicate=False,
        min_text_chars=10,
    )
    out = tmp_path / "build"
    DatasetBuilder(config).build(SplitConfig(seed=42), target=LabelLevel.DIALECT, output_dir=out)
    return read_json(out / "build_manifest.json")


def _trained_report(tmp_path: Path) -> tuple[dict, EvaluationReport]:
    """Train + save a tiny classifier; return its sidecar and a test report."""
    samples = make_samples()
    classifier = DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42)
    classifier.fit(samples)
    _, sidecar = load_model(classifier.save(tmp_path / "model"))

    batch = classifier.labelled_batch(samples)
    proba = classifier.predict_proba(batch.raws)
    y_pred = [classifier.classes_[int(i)] for i in np.argmax(proba, axis=1)]
    labels = sorted(set(classifier.classes_) | set(batch.labels))
    report = compute_metrics(
        batch.labels,
        y_pred,
        y_proba=proba,
        labels=labels,
        metadata={"model": "logistic_regression", "split": "test"},
    )
    return sidecar, report


# ------------------------------------------------------------- dataset card


def test_dataset_card_from_manifest_contains_key_facts(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    card = dataset_card(_INFO, manifest)

    assert card.startswith("# Dataset card — synthetic")
    assert "A tiny synthetic dialect corpus." in card
    assert "https://example.org/synthetic" in card
    assert "CC0-1.0" in card

    # Class names are humanised via the taxonomy, not shown as raw slugs.
    observed = {lab for counts in manifest["class_distribution"].values() for lab in counts}
    assert observed  # the corpus is labelled at dialect level
    for label in observed:
        assert display_name(label) in card
    # Source corpora reuse the manifest's recorded counts (not recomputed).
    for source in manifest["sources"]:
        assert source in card
    # Total sample count reused from the manifest; speakers unknown -> n/a.
    assert f"| **Total** | {manifest['total']} | n/a |" in card


def test_dataset_card_from_splits_computes_speaker_counts() -> None:
    splits = speaker_disjoint_split(make_samples(repeats=4), SplitConfig(seed=42))
    card = dataset_card_from_splits(_INFO, splits)

    # 4 dialects x 4 speakers x 4 sentences = 64 samples across 16 distinct speakers.
    assert "| **Total** | 64 | 16 |" in card
    assert "## Splits" in card
    # Dialect labels are humanised; standard samples (dialect=None) fall to the
    # unlabelled sentinel at the dialect level.
    assert "Podhale" in card
    assert "(unlabelled)" in card


def test_dataset_card_is_byte_stable(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    assert dataset_card(_INFO, manifest) == dataset_card(_INFO, manifest)

    splits = speaker_disjoint_split(make_samples(repeats=4), SplitConfig(seed=42))
    assert dataset_card_from_splits(_INFO, splits) == dataset_card_from_splits(_INFO, splits)


def test_dataset_card_degrades_gracefully_on_empty_manifest() -> None:
    card = dataset_card(DatasetInfo(name="bare"), {})
    assert card.startswith("# Dataset card — bare")
    assert "n/a" in card  # empty url/tasks/label-levels render as n/a
    assert "No class distribution recorded." in card
    assert "No source corpora recorded." in card
    assert "## Splits" in card  # a totals-only split table still renders


# --------------------------------------------------------------- model card


def test_model_card_contains_key_facts(tmp_path: Path) -> None:
    sidecar, report = _trained_report(tmp_path)
    card = model_card(sidecar, {"test": report})

    assert card.startswith("# Model card — logistic_regression")
    assert sidecar["model_class"] in card
    assert sidecar["tulip_version"] in card
    assert sidecar["python_version"] in card
    assert "**Task:** text" in card
    assert "**Target level:** dialect" in card
    assert "char_tfidf" in card  # feature component listed

    # Fitted classes appear with both raw slug and humanised name.
    for label in sidecar["classes"]:
        assert f"`{label}`" in card
        assert display_name(label) in card

    # Headline metrics table keyed by split, plus the report's summary line.
    assert "## Metrics" in card
    assert "| Metric | test |" in card
    assert f"| Samples | {report.n_samples} |" in card
    assert "- **test:** " in card


def test_model_card_is_byte_stable(tmp_path: Path) -> None:
    sidecar, report = _trained_report(tmp_path)
    reports = {"validation": report, "test": report}
    assert model_card(sidecar, reports) == model_card(sidecar, reports)


def test_model_card_degrades_gracefully_on_empty_sidecar() -> None:
    card = model_card({}, {})
    assert card.startswith("# Model card — model")
    assert "Raw-input model (no feature components)." in card
    assert "Classes: n/a" in card
    assert "No evaluation reports available." in card
    assert "No non-default parameters." in card
