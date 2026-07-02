"""Shared fixtures for the tulip test suite."""

from __future__ import annotations

import numpy as np
import pytest

from tulip.core.types import DialectLabels, Sample

# Tiny synthetic corpus: crude dialect-flavoured Polish, multiple speakers per
# dialect so speaker-disjoint splitting is exercisable. Not linguistically
# faithful; only the statistical shape matters for tests.
_DIALECT_SENTENCES: dict[str, list[str]] = {
    "podhale": [
        "Hej, baca się pyto, kaj się owce pasą na holi.",
        "Juhas poseł na grań i widzioł cołkiem piykne hole.",
        "Kie góral idzie ku dolinie, to se śpiywo po naszymu.",
        "Baca wzion ciupagę i poseł ku sałasowi na wiyrchu.",
    ],
    "silesia": [
        "Jo żech je z Katowic i godom po naszymu cołki czos.",
        "Kaj żeś boł wczorej, bo cie żech nie widzioł na placu?",
        "Dej pozór na bajtla, bo lecie po drodze ku familokowi.",
        "Niy ma co godać, trza robić, pado starzik.",
    ],
    "kurpie": [
        "U nos w boru to psiwo warzą jesce po staremu.",
        "Chłopoki poślo do lasu na jagody i grziby zbzierać.",
        "Kobziety śpsiewajo w kościele w niedziele rano.",
        "Na Kurpsiach bursztyn kopalo się od dawna w psiochach.",
    ],
    "standard": [
        "Wczoraj pojechałem do Warszawy na spotkanie służbowe.",
        "Prognoza pogody zapowiada deszcz w całym kraju.",
        "Nowa ustawa wejdzie w życie od pierwszego stycznia.",
        "Dzieci wróciły ze szkoły i odrobiły lekcje przed kolacją.",
    ],
}


def make_samples(*, repeats: int = 3) -> list[Sample]:
    """Build a small deterministic corpus with several speakers per dialect."""
    samples: list[Sample] = []
    for dialect, sentences in _DIALECT_SENTENCES.items():
        for speaker_index in range(repeats):
            speaker = f"{dialect}-spk{speaker_index}"
            for sentence_index, sentence in enumerate(sentences):
                samples.append(
                    Sample(
                        id=f"{speaker}-{sentence_index}",
                        text=f"{sentence} (wariant {speaker_index})",
                        speaker_id=speaker,
                        labels=DialectLabels(
                            dialect=dialect if dialect != "standard" else None,
                            family="standard" if dialect == "standard" else None,
                        ),
                        source="synthetic",
                    )
                )
    return samples


@pytest.fixture
def synthetic_samples() -> list[Sample]:
    """A small labelled corpus spanning four classes and twelve speakers."""
    return make_samples()


@pytest.fixture
def synthetic_texts_and_labels(synthetic_samples: list[Sample]) -> tuple[list[str], list[str]]:
    """Parallel lists of texts and dialect-or-standard labels."""
    texts = [s.text or "" for s in synthetic_samples]
    labels = [s.labels.dialect or s.labels.family or "unknown" for s in synthetic_samples]
    return texts, labels


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded numpy random generator for deterministic tests."""
    return np.random.default_rng(42)
