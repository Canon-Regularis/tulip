"""Tests for the generated synthetic reference corpus.

The corpus is the toolkit's zero-acquisition escape hatch, so these tests guard
the two properties that make it worth having: it is *reproducible* (a seed fixes
the output byte for byte) and it carries *real, learnable signal* rather than
noise. The learnability guard is the load-bearing one -- a synthetic corpus that
trains to chance would make every downstream benchmark meaningless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.config.schemas import SplitConfig
from tulip.core.exceptions import ConfigurationError
from tulip.data.dedup import deduplicate_samples
from tulip.data.loaders.synthetic import SyntheticLoader
from tulip.data.registry import DATASETS
from tulip.data.splitting import speaker_disjoint_split
from tulip.data.synthetic import (
    SOURCE,
    SyntheticSpec,
    generate_corpus,
    write_synthetic_manifest,
)
from tulip.labels.taxonomy import DialectFamily, RegionalDialect, family_for

if TYPE_CHECKING:
    from pathlib import Path

    from tulip.core.types import Sample

#: Small, fast spec for structural assertions.
SMALL = SyntheticSpec(n_speakers_per_dialect=3, samples_per_speaker=4, seed=7)


def _class_of(sample: Sample) -> str | None:
    """The sample's dialect, or its family for the standard negative class."""
    return sample.labels.dialect or sample.labels.family


# ------------------------------------------------------------- determinism


def test_same_seed_yields_identical_corpus() -> None:
    first = generate_corpus(SMALL)
    second = generate_corpus(SMALL)
    assert [s.id for s in first] == [s.id for s in second]
    assert [s.text for s in first] == [s.text for s in second]
    assert [_class_of(s) for s in first] == [_class_of(s) for s in second]


def test_different_seed_yields_different_text() -> None:
    other = generate_corpus(SyntheticSpec(n_speakers_per_dialect=3, samples_per_speaker=4, seed=8))
    assert [s.text for s in generate_corpus(SMALL)] != [s.text for s in other]


# ------------------------------------------------------------------ labels


def test_dialect_labels_are_taxonomy_values_and_families_derive() -> None:
    dialect_values = {d.value for d in RegionalDialect}
    for sample in generate_corpus(SMALL):
        if sample.labels.dialect is None:
            continue
        assert sample.labels.dialect in dialect_values
        # DialectLabels auto-derives family; it must agree with the taxonomy.
        assert sample.labels.family == family_for(sample.labels.dialect).value


def test_masovia_lexicon_key_maps_to_mazovia_proper() -> None:
    """The lexicon groups Masovian markers under a key that is not a taxonomy value.

    ``family_for("masovia")`` is ``None``; only ``mazovia_proper`` resolves. A
    regression here would silently produce family-less Masovian samples.
    """
    corpus = generate_corpus(SyntheticSpec(dialects=("masovia",), include_standard=False, seed=7))
    assert {s.labels.dialect for s in corpus} == {RegionalDialect.MAZOVIA_PROPER.value}
    assert {s.labels.family for s in corpus} == {DialectFamily.MASOVIAN.value}


def test_standard_class_carries_family_only() -> None:
    standard = [s for s in generate_corpus(SMALL) if s.labels.family == "standard"]
    assert standard, "include_standard=True must emit a standard negative class"
    assert all(s.labels.dialect is None for s in standard)


def test_every_class_has_at_least_two_speakers() -> None:
    by_class: dict[str, set[str]] = {}
    for sample in generate_corpus(SMALL):
        by_class.setdefault(_class_of(sample), set()).add(sample.speaker_id)
    assert by_class, "corpus must not be empty"
    assert all(len(speakers) >= 2 for speakers in by_class.values())


def test_samples_are_sourced_and_marked_as_generated() -> None:
    for sample in generate_corpus(SMALL):
        assert sample.source == SOURCE
        assert sample.metadata["generator"] == "tulip-synthetic"
        assert sample.metadata["spec_seed"] == SMALL.seed


# --------------------------------------------------- pipeline compatibility


def test_texts_clear_the_default_min_text_chars() -> None:
    """DataConfig.min_text_chars defaults to 20; shorter texts would be dropped."""
    assert min(len(s.text) for s in generate_corpus(SMALL)) >= 20


def test_dedup_keeps_the_corpus_and_never_empties_a_class() -> None:
    """Near-dup collapse would silently delete whole classes before splitting.

    Benchmark configs keep ``deduplicate: true``, so the carrier slot-filling
    must produce texts varied enough to survive the char-shingle Jaccard pass.
    """
    corpus = generate_corpus(SyntheticSpec(n_speakers_per_dialect=6, samples_per_speaker=8, seed=7))
    kept = deduplicate_samples(corpus).samples
    assert len(kept) >= 0.95 * len(corpus)
    survivors = {_class_of(s) for s in kept}
    assert survivors == {_class_of(s) for s in corpus}


def test_speaker_disjoint_split_succeeds_and_leaks_no_speaker() -> None:
    corpus = generate_corpus(SyntheticSpec(n_speakers_per_dialect=8, samples_per_speaker=6, seed=7))
    splits = speaker_disjoint_split(corpus, SplitConfig(seed=42))
    groups = [{s.speaker_id for s in part} for part in splits.as_dict().values()]
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            assert not (left & right)


# ------------------------------------------------------- signal, not noise


@pytest.mark.parametrize("model_name", ["logistic_regression"])
def test_corpus_is_learnable_well_above_chance(model_name: str) -> None:
    """The guard that makes the whole corpus worth shipping.

    Trains on a speaker-disjoint split, so the score cannot come from speaker
    re-identification. Chance is 1/6; a healthy corpus lands far above it.
    """
    from tulip.pipeline.classifier import DialectClassifier

    corpus = generate_corpus(
        SyntheticSpec(
            n_speakers_per_dialect=12, samples_per_speaker=12, include_standard=False, seed=7
        )
    )
    splits = speaker_disjoint_split(corpus, SplitConfig(seed=42))
    classifier = DialectClassifier(model=model_name, features=["char_tfidf"]).fit(splits.train)

    predictions = classifier.predict_batch([s.text for s in splits.test])
    gold = [s.labels.at_level(classifier.target) for s in splits.test]
    accuracy = sum(p.label == g for p, g in zip(predictions, gold, strict=True)) / len(gold)

    assert accuracy > 0.6, f"synthetic corpus lost its signal: accuracy={accuracy:.3f}"


def test_full_marker_dropout_removes_lexical_markers() -> None:
    """With dropout=1.0 a class with no phonological transform loses its cues.

    Podhale has marker lexemes but no sound change, so every marker must vanish.
    This pins the knob that gives the benchmark its irreducible error floor.
    """
    corpus = generate_corpus(
        SyntheticSpec(
            dialects=("podhale",),
            include_standard=False,
            marker_dropout=1.0,
            noise_level=0.0,
            n_speakers_per_dialect=4,
            samples_per_speaker=6,
            seed=7,
        )
    )
    markers = {"baca", "juhas", "ciupaga", "watra", "gazda", "dutki"}
    for sample in corpus:
        assert not markers & set(sample.text.lower().split())


# ------------------------------------------------------------------ loader


def test_loader_is_registered_and_generates_on_demand(tmp_path: Path) -> None:
    assert "synthetic" in DATASETS.names()
    loader = DATASETS.create("synthetic", n_speakers_per_dialect=2, samples_per_speaker=2)
    samples = list(loader.load(tmp_path))  # tmp_path holds no manifest
    assert samples and loader.is_available(tmp_path)


def test_written_manifest_round_trips_through_the_loader(tmp_path: Path) -> None:
    spec = SyntheticSpec(n_speakers_per_dialect=2, samples_per_speaker=3, seed=7)
    path = write_synthetic_manifest(spec, tmp_path)
    assert path.is_file()

    reread = list(SyntheticLoader(manifest="manifest.jsonl").load(tmp_path))
    original = generate_corpus(spec)
    assert [s.id for s in reread] == [s.id for s in original]
    assert [s.text for s in reread] == [s.text for s in original]
    assert [_class_of(s) for s in reread] == [_class_of(s) for s in original]


def test_loader_raises_when_a_configured_manifest_is_absent(tmp_path: Path) -> None:
    from tulip.core.exceptions import DataError

    with pytest.raises(DataError, match="manifest not found"):
        list(SyntheticLoader(manifest="missing.jsonl").load(tmp_path))


def test_read_samples_preserves_labels_of_a_jsonl_manifest(tmp_path: Path) -> None:
    """A JSONL manifest must not be misread as a split file.

    Split files and manifests share the ``.jsonl`` suffix but differ in shape:
    a manifest's label columns are flat. Such a record still validates as a
    label-less ``Sample``, so a naive reader silently drops every label and
    downstream evaluation scores against nothing.
    """
    from tulip.data.reading import read_samples

    spec = SyntheticSpec(n_speakers_per_dialect=2, samples_per_speaker=2, seed=7)
    path = write_synthetic_manifest(spec, tmp_path)

    labels = {_class_of(sample) for sample in read_samples(path)}
    assert labels == {_class_of(sample) for sample in generate_corpus(spec)}
    assert "__unlabelled__" not in labels


# -------------------------------------------------------------- spec guards


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_speakers_per_dialect": 1},
        {"samples_per_speaker": 0},
        {"noise_level": 1.5},
        {"marker_dropout": -0.1},
        {"dialects": ()},
    ],
)
def test_spec_rejects_out_of_range_knobs(kwargs: dict) -> None:
    with pytest.raises(ConfigurationError):
        SyntheticSpec(**kwargs)


def test_unknown_dialect_key_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown synthetic dialect"):
        generate_corpus(SyntheticSpec(dialects=("atlantis",)))
