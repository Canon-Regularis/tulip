"""Loader for Dialektarium (https://dialektarium.pl/)."""

from __future__ import annotations

from typing import ClassVar

from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.registry import DATASETS


@DATASETS.register("dialektarium")
class DialektariumLoader(ManifestBackedLoader):
    """Dialektarium: dialect recordings with transcriptions and geography.

    Tier-1 corpus: audio, aligned transcriptions, and per-sample village and
    region metadata. There is no bulk download; assemble the corpus locally
    (see ``docs/datasets.md``) into::

        data/raw/dialektarium/
            manifest.csv          # or .tsv / .jsonl
            clips/<file>.wav      # audio, referenced relatively from the manifest

    Expected manifest columns (standard names; all optional except at least
    one of ``text``/``audio_path``): ``id``, ``text``, ``audio_path``,
    ``speaker_id``, ``dialect``, ``region``, ``village``, ``voivodeship``.
    When ``speaker_id`` is absent, a stable surrogate is synthesised from the
    village/region metadata, so recordings from one locality group together
    for speaker-disjoint splitting.
    """

    dataset_name = "dialektarium"
    acquisition: ClassVar[str] = (
        "manual: no bulk download exists; export or transcribe material you are "
        "licensed to use from https://dialektarium.pl/ into "
        "data/raw/dialektarium/manifest.csv + clips/ (see docs/datasets.md)"
    )


__all__ = ["DialektariumLoader"]
