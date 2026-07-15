"""Declarative catalog of the source corpora tulip knows how to load.

Each corpus is described by a frozen :class:`~tulip.core.types.DatasetInfo`
record: canonical name, acquisition URL, quality tier, supported tasks, and
the label granularities its metadata can populate. Tiers rank how directly a
corpus serves dialect classification:

* **Tier 1**: dialect corpora with fine-grained geographic metadata.
* **Tier 2**: single-dialect corpora (positive examples for one class).
* **Tier 3**: large general-Polish corpora (standard-language negatives,
  or dialect signal only via self-reported accent metadata).
* **Tier 4**: aggregated ASR benchmarks; useful audio, weak dialect labels.

The catalog is intentionally static and import-cheap: acquisition of every
corpus is documented in ``docs/datasets.md`` and tulip never scrapes remote
sources at runtime.
"""

from __future__ import annotations

from tulip.core.exceptions import DataError
from tulip.core.types import DatasetInfo
from tulip.labels.taxonomy import LabelLevel

_TEXT = "text"
_AUDIO = "audio"

_CATALOG: dict[str, DatasetInfo] = {
    info.name: info
    for info in (
        DatasetInfo(
            name="dialektarium",
            description=(
                "Dialektarium: recordings of dialectal Polish with aligned "
                "transcriptions and per-sample village/region metadata."
            ),
            url="https://dialektarium.pl/",
            tier=1,
            tasks=(_TEXT, _AUDIO),
            contents=("audio", "transcriptions", "village/region metadata"),
            label_levels=(
                LabelLevel.FAMILY,
                LabelLevel.DIALECT,
                LabelLevel.REGION,
                LabelLevel.VILLAGE,
            ),
        ),
        DatasetInfo(
            name="dgp",
            description=(
                "Dialekty i gwary polskie. Kompendium internetowe: curated "
                "dialect text samples organised by dialect group and region."
            ),
            url="https://przewodnik.tmjp.pl/dgp-dialekty-i-gwary-polskie-kompendium-internetowe/",
            tier=1,
            tasks=(_TEXT,),
            contents=("dialect text samples", "dialect/region/village metadata"),
            label_levels=(
                LabelLevel.FAMILY,
                LabelLevel.DIALECT,
                LabelLevel.REGION,
                LabelLevel.VILLAGE,
            ),
        ),
        DatasetInfo(
            name="korpus_spiski",
            description=(
                "Korpus Spiski: a corpus of the Spisz dialect (spoken-language "
                "transcriptions from Spisz villages)."
            ),
            url="https://journals.akademicka.pl/lv/article/view/727",
            tier=2,
            tasks=(_TEXT,),
            contents=("dialect transcriptions", "village metadata"),
            label_levels=(LabelLevel.FAMILY, LabelLevel.DIALECT, LabelLevel.VILLAGE),
        ),
        DatasetInfo(
            name="mackowce",
            description=(
                "Elektroniczny Korpus Tekstow Gwarowych z Mackowiec: dialect "
                "texts from Mackowce in Podolia (Polish borderland variety)."
            ),
            url=(
                "https://przewodnik.tmjp.pl/"
                "ektgm-elektroniczny-korpus-tekstow-gwarowych-z-mackowiec-na-podolu/"
            ),
            tier=2,
            tasks=(_TEXT,),
            contents=("dialect texts", "speaker metadata"),
            label_levels=(LabelLevel.FAMILY, LabelLevel.DIALECT, LabelLevel.VILLAGE),
        ),
        DatasetInfo(
            name="nkjp",
            description=(
                "Narodowy Korpus Jezyka Polskiego: standard (general) Polish; "
                "used as negative examples for dialect-vs-standard tasks."
            ),
            url="https://nkjp.pl/",
            tier=3,
            tasks=(_TEXT,),
            contents=("standard Polish text",),
            label_levels=(LabelLevel.FAMILY,),
        ),
        DatasetInfo(
            name="spokes",
            description=(
                "Spokes: conversational spoken Polish (transcribed spontaneous "
                "speech); predominantly standard Polish."
            ),
            url="https://spokes.clarin-pl.eu/",
            tier=3,
            tasks=(_TEXT,),
            contents=("conversational transcriptions", "speaker metadata"),
            label_levels=(LabelLevel.FAMILY,),
        ),
        DatasetInfo(
            name="common_voice_pl",
            description=(
                "Mozilla Common Voice (Polish): crowd-read speech with "
                "sentence transcripts and self-reported accent metadata."
            ),
            url="https://commonvoice.mozilla.org/",
            tier=3,
            tasks=(_TEXT, _AUDIO),
            contents=("audio clips", "sentence transcripts", "accent/variant metadata"),
            label_levels=(LabelLevel.FAMILY, LabelLevel.DIALECT),
            license="CC0-1.0",
        ),
        DatasetInfo(
            name="bigos",
            description=(
                "BIGOS: an aggregated benchmark of Polish ASR corpora "
                "(audio + transcriptions from many source datasets)."
            ),
            url="https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos",
            tier=4,
            tasks=(_TEXT, _AUDIO),
            contents=("audio", "transcriptions", "source-corpus metadata"),
            label_levels=(),
            license="varies per source subset; see the dataset card",
        ),
        # Tier 4 because the frozen DatasetInfo bound is 1..4, so a dedicated
        # "generated" tier cannot be expressed. Keep that friction here, in a
        # comment: `description` is rendered verbatim into user-facing dataset
        # cards and must describe the corpus, not our type constraints.
        DatasetInfo(
            name="synthetic",
            description=(
                "Synthetic reference corpus: procedurally generated Polish dialect "
                "text with injected lexical markers and phonological transforms "
                "(mazurzenie, asynchronous soft-labials). Generated in-process, so "
                "the toolkit runs end-to-end with zero data acquisition. It is a "
                "reproducible benchmark fixture, NOT real speech: scores on it say "
                "nothing about real-world dialect identification accuracy."
            ),
            url="generated in-process; see tulip.data.synthetic and docs/datasets.md",
            tier=4,
            tasks=(_TEXT,),
            contents=(
                "generated dialect text",
                "injected lexical markers",
                "phonological transforms",
                "region/voivodeship metadata",
            ),
            label_levels=(
                LabelLevel.FAMILY,
                LabelLevel.DIALECT,
                LabelLevel.REGION,
                LabelLevel.VOIVODESHIP,
            ),
            license="generated data; no source-corpus restrictions (public domain)",
        ),
        # Tier 4 for the same reason as "synthetic": the frozen DatasetInfo tier
        # bound is 1..4, so a dedicated "generated" tier cannot be expressed.
        # Implementation notes (source-filter synthesis, per-class F0/formants)
        # live in tulip.data.synthetic_audio, NOT here: `description` is rendered
        # verbatim into user-facing dataset cards and must describe the corpus.
        DatasetInfo(
            name="synthetic_audio",
            description=(
                "Synthetic audio reference corpus: procedurally synthesised speech-like "
                "WAV clips whose pitch register, vowel-space formants, and spectral tilt "
                "are correlated with dialect class. Generated in-process, so the audio "
                "path runs end-to-end with zero data acquisition. It is a reproducible "
                "benchmark fixture, NOT real speech: scores on it say nothing about "
                "real-world dialect identification accuracy."
            ),
            url="generated in-process; see tulip.data.synthetic_audio and docs/datasets.md",
            tier=4,
            tasks=(_AUDIO,),
            contents=(
                "generated speech-like audio",
                "dialect-correlated F0/formants/spectral tilt",
                "region/voivodeship metadata",
            ),
            label_levels=(
                LabelLevel.FAMILY,
                LabelLevel.DIALECT,
                LabelLevel.REGION,
                LabelLevel.VOIVODESHIP,
            ),
            license="generated data; no source-corpus restrictions (public domain)",
        ),
    )
}


def catalog() -> list[DatasetInfo]:
    """Return all catalogued corpora, sorted by tier then name.

    Returns:
        Every :class:`DatasetInfo` in the catalog, most useful tiers first.
    """
    return sorted(_CATALOG.values(), key=lambda info: (info.tier, info.name))


def get_dataset_info(name: str) -> DatasetInfo:
    """Return the catalog entry for ``name``.

    Args:
        name: Canonical corpus name (e.g. ``"dialektarium"``).

    Raises:
        DataError: if ``name`` is not in the catalog.
    """
    key = name.strip().lower()
    try:
        return _CATALOG[key]
    except KeyError:
        known = ", ".join(sorted(_CATALOG))
        raise DataError(f"unknown dataset {name!r}; catalogued datasets: {known}") from None


__all__ = ["catalog", "get_dataset_info"]
