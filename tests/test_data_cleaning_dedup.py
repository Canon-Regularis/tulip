"""Tests for text cleaning and deduplication."""

from __future__ import annotations

from tulip.core.types import DialectLabels, Sample
from tulip.data.cleaning import TextCleaner
from tulip.data.dedup import deduplicate_samples, shingle_jaccard


def _sample(sample_id: str, text: str | None, speaker: str = "spk-1") -> Sample:
    return Sample(
        id=sample_id,
        text=text,
        audio_path=None if text is not None else "clip.wav",  # type: ignore[arg-type]
        speaker_id=speaker,
        labels=DialectLabels(dialect="podhale"),
        source="test",
    )


class TestTextCleaner:
    def test_removes_bracketed_annotations(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.clean("No i [śmiech] tego było {pauza} tak <laugh>") == "No i tego było tak"

    def test_removes_only_annotation_parentheses(self) -> None:
        cleaner = TextCleaner()
        cleaned = cleaner.clean("Poszedł (niezrozumiałe) do lasu (za wsią)")
        assert "(niezrozumiałe)" not in cleaned
        assert "(za wsią)" in cleaned  # ordinary prose parentheses survive

    def test_normalises_typographic_punctuation(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.clean("„Hej” — rzekł…") == '"Hej" - rzekł...'

    def test_preserves_dialectal_orthography(self) -> None:
        cleaner = TextCleaner(lowercase=True)
        # Diacritics and non-standard spellings must survive every step.
        assert cleaner.clean("Kaj żeś BOŁ? Psiwo, gaździna!") == "kaj żeś boł? psiwo, gaździna!"

    def test_collapses_whitespace_and_squeezes_punctuation(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.clean("  No   i??  co,,  teraz  ") == "No i? co, teraz"

    def test_steps_are_toggleable(self) -> None:
        cleaner = TextCleaner(remove_artifacts=False, normalise_punctuation=False)
        assert cleaner.clean("Hej  [śmiech] —") == "Hej [śmiech] —"

    def test_clean_sample_returns_same_object_when_unchanged(self) -> None:
        cleaner = TextCleaner()
        sample = _sample("s1", "Czysty tekst bez artefaktów.")
        assert cleaner.clean_sample(sample) is sample

    def test_config_round_trips_flags(self) -> None:
        cleaner = TextCleaner(lowercase=True, nfc=False)
        assert cleaner.config()["lowercase"] is True
        assert cleaner.config()["nfc"] is False


LONG_A = (
    "Kie baca poseł na grań, to widzioł całkiem piykne hole i pasące się owce, "
    "a juhasi śpiywali po naszymu przy watrze do samego rana."
)
LONG_B = (
    "Kie baca poseł na grań, to widzioł całkiem piykne hole i pasące się kozy, "
    "a juhasi śpiywali po naszymu przy watrze do samego rana."
)
LONG_C = (
    "Prognoza pogody na jutro zapowiada przelotne opady deszczu w całym kraju "
    "oraz silny wiatr na wybrzeżu i w górach, miejscami burze."
)


class TestDeduplication:
    def test_exact_duplicates_are_dropped_keeping_first(self) -> None:
        result = deduplicate_samples(
            [_sample("a", LONG_A), _sample("b", LONG_A), _sample("c", LONG_C)]
        )
        assert [s.id for s in result.samples] == ["a", "c"]
        assert result.dropped_exact == ["b"]

    def test_exact_pass_ignores_case_and_whitespace_noise(self) -> None:
        result = deduplicate_samples(
            [_sample("a", LONG_A), _sample("b", "  " + LONG_A.upper() + "  ")],
            near_duplicates=False,
        )
        assert result.dropped_exact == ["b"]

    def test_near_duplicates_are_dropped(self) -> None:
        assert shingle_jaccard(LONG_A, LONG_B) > 0.85
        result = deduplicate_samples([_sample("a", LONG_A), _sample("b", LONG_B)])
        assert [s.id for s in result.samples] == ["a"]
        assert result.dropped_near == ["b"]

    def test_distinct_texts_survive(self) -> None:
        result = deduplicate_samples([_sample("a", LONG_A), _sample("c", LONG_C)])
        assert result.num_dropped == 0

    def test_near_pass_can_be_disabled(self) -> None:
        result = deduplicate_samples(
            [_sample("a", LONG_A), _sample("b", LONG_B)], near_duplicates=False
        )
        assert result.num_dropped == 0

    def test_audio_only_samples_always_survive(self) -> None:
        result = deduplicate_samples([_sample("a", None), _sample("b", None)])
        assert len(result.samples) == 2

    def test_deterministic_across_runs(self) -> None:
        samples = [_sample(f"s{i}", f"{LONG_A} wariant {i % 3}") for i in range(30)]
        first = deduplicate_samples(samples)
        second = deduplicate_samples(samples)
        assert [s.id for s in first.samples] == [s.id for s in second.samples]
