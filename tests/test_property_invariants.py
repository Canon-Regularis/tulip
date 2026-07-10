"""Property-based tests for tulip's load-bearing invariants.

Example-based tests pin the cases we thought of. These pin the *contract*: for
every input the API accepts, the guarantee holds. Three of these guard against
silent data corruption rather than crashes -- speaker leakage across splits,
samples vanishing during splitting, and non-deterministic deduplication would
all produce a benchmark that looks fine and reports inflated scores.

Each property either holds or the documented ``DataError`` is raised. Nothing
here catches a bare ``Exception``: a property test that swallows failures proves
nothing at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tulip.config.schemas import SplitConfig
from tulip.core.exceptions import DataError
from tulip.core.types import DialectLabels, Sample
from tulip.data.dedup import deduplicate_samples
from tulip.data.splitting import speaker_disjoint_split
from tulip.evaluation.metrics import compute_metrics

#: Real taxonomy dialects, so DialectLabels' family derivation exercises too.
_DIALECTS = ("podhale", "silesia", "kashubia", "kurpie")

#: Long enough to survive shingling (dedup uses 5-char shingles).
_TEXTS = st.text(alphabet="abcdefghijklmnop ", min_size=12, max_size=60)

# The suite runs on four Pythons in CI; keep the budget modest but meaningful.
_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@st.composite
def corpora(draw: st.DrawFn, *, min_speakers: int = 1, max_speakers: int = 8) -> list[Sample]:
    """Generate a corpus with explicit speakers and dialect labels."""
    n_speakers = draw(st.integers(min_value=min_speakers, max_value=max_speakers))
    samples: list[Sample] = []
    for speaker in range(n_speakers):
        dialect = draw(st.sampled_from(_DIALECTS))
        for index in range(draw(st.integers(min_value=1, max_value=4))):
            samples.append(
                Sample(
                    id=f"spk{speaker}-{index}",
                    text=draw(_TEXTS),
                    speaker_id=f"spk{speaker}",
                    labels=DialectLabels(dialect=dialect),
                    source="property-test",
                )
            )
    return samples


# ----------------------------------------------------------------- splitting


@_SETTINGS
@given(samples=corpora(), seed=st.integers(min_value=0, max_value=9999))
def test_splits_never_share_a_speaker(samples: list[Sample], seed: int) -> None:
    """The anti-leakage guarantee: no speaker may appear in two splits.

    A violation lets a classifier re-identify the speaker instead of the
    dialect, which inflates every downstream benchmark number.
    """
    try:
        splits = speaker_disjoint_split(samples, SplitConfig(seed=seed))
    except DataError:
        return  # documented: raise rather than emit an empty split

    parts = [{s.speaker_id for s in part} for part in splits.as_dict().values()]
    for i, left in enumerate(parts):
        for right in parts[i + 1 :]:
            assert not (left & right)


@_SETTINGS
@given(samples=corpora(), seed=st.integers(min_value=0, max_value=9999))
def test_splitting_loses_and_duplicates_nothing(samples: list[Sample], seed: int) -> None:
    """Every sample lands in exactly one split, or DataError is raised."""
    try:
        splits = speaker_disjoint_split(samples, SplitConfig(seed=seed))
    except DataError:
        return

    placed = [s.id for part in splits.as_dict().values() for s in part]
    assert sorted(placed) == sorted(s.id for s in samples)
    assert len(placed) == len(set(placed))  # no duplication


@_SETTINGS
@given(samples=corpora(), seed=st.integers(min_value=0, max_value=9999))
def test_splitting_is_deterministic_under_a_fixed_seed(samples: list[Sample], seed: int) -> None:
    try:
        first = speaker_disjoint_split(samples, SplitConfig(seed=seed))
        second = speaker_disjoint_split(samples, SplitConfig(seed=seed))
    except DataError:
        return

    for name, part in first.as_dict().items():
        assert [s.id for s in part] == [s.id for s in second.as_dict()[name]]


@_SETTINGS
@given(samples=corpora(min_speakers=8, max_speakers=12), seed=st.integers(0, 9999))
def test_a_corpus_with_ample_speakers_actually_splits(samples: list[Sample], seed: int) -> None:
    """Guards the properties above from passing vacuously via the DataError path.

    With this many distinct speakers a 70/15/15 split must succeed, so the
    ``except DataError: return`` branches cannot be hiding a broken splitter.
    """
    splits = speaker_disjoint_split(samples, SplitConfig(seed=seed))
    assert splits.total == len(samples)
    assert all(size > 0 for size in splits.sizes().values())


# ---------------------------------------------------------------------- dedup


@_SETTINGS
@given(samples=corpora())
def test_dedup_is_deterministic_and_a_fixed_point(samples: list[Sample]) -> None:
    """Re-running dedup on its own output must drop nothing."""
    once = deduplicate_samples(samples).samples
    again = deduplicate_samples(samples).samples
    assert [s.id for s in once] == [s.id for s in again]  # deterministic

    twice = deduplicate_samples(once).samples
    assert [s.id for s in twice] == [s.id for s in once]  # idempotent


@_SETTINGS
@given(samples=corpora())
def test_dedup_never_grows_or_empties_a_non_empty_corpus(samples: list[Sample]) -> None:
    kept = deduplicate_samples(samples).samples
    assert 0 < len(kept) <= len(samples)


@_SETTINGS
@given(text=_TEXTS, copies=st.integers(min_value=2, max_value=6))
def test_identical_texts_always_collapse_to_one(text: str, copies: int) -> None:
    """Exact duplicates must never straddle a split, so they collapse first."""
    samples = [
        Sample(
            id=f"dup-{i}",
            text=text,
            speaker_id=f"spk{i}",
            labels=DialectLabels(dialect="podhale"),
            source="property-test",
        )
        for i in range(copies)
    ]
    result = deduplicate_samples(samples)
    assert len(result.samples) == 1
    assert result.samples[0].id == "dup-0"  # first occurrence wins
    assert result.num_dropped == copies - 1


# -------------------------------------------------------------------- metrics


@st.composite
def label_pairs(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    """Aligned (y_true, y_pred) over a shared label set."""
    labels = draw(st.lists(st.sampled_from(_DIALECTS), min_size=2, max_size=4, unique=True))
    size = draw(st.integers(min_value=2, max_value=30))
    y_true = draw(st.lists(st.sampled_from(labels), min_size=size, max_size=size))
    y_pred = draw(st.lists(st.sampled_from(labels), min_size=size, max_size=size))
    assume(len(set(y_true)) >= 2)  # a degenerate single-class split is not the subject
    return y_true, y_pred


@_SETTINGS
@given(pair=label_pairs())
def test_metrics_stay_within_their_declared_ranges(pair: tuple[list[str], list[str]]) -> None:
    y_true, y_pred = pair
    report = compute_metrics(y_true, y_pred)

    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.f1_macro <= 1.0
    for metrics in report.per_class.values():
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
    assert report.n_samples == len(y_true)


@_SETTINGS
@given(pair=label_pairs())
def test_a_perfect_prediction_scores_exactly_one(pair: tuple[list[str], list[str]]) -> None:
    y_true, _ = pair
    report = compute_metrics(y_true, y_true)
    assert report.accuracy == pytest.approx(1.0)
    assert report.f1_macro == pytest.approx(1.0)


@_SETTINGS
@given(pair=label_pairs())
def test_confusion_row_sums_equal_the_true_class_counts(pair: tuple[list[str], list[str]]) -> None:
    """``confusion[i][j]`` counts true ``labels[i]`` predicted as ``labels[j]``."""
    y_true, y_pred = pair
    report = compute_metrics(y_true, y_pred)

    matrix = np.asarray(report.confusion)
    for index, label in enumerate(report.labels):
        assert int(matrix[index].sum()) == y_true.count(label)
    assert int(matrix.sum()) == len(y_true)
