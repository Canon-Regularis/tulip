"""Loader for Spokes, the conversational spoken-Polish corpus."""

from __future__ import annotations

from typing import ClassVar

from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.registry import DATASETS


@DATASETS.register("spokes")
class SpokesLoader(ManifestBackedLoader):
    """Spokes (https://spokes.clarin-pl.eu/): conversational transcripts.

    Tier-3 corpus of transcribed spontaneous conversations, predominantly
    standard (colloquial) Polish -- useful as spoken-register negatives that
    match dialect corpora in genre. Export locally (see ``docs/datasets.md``)
    into::

        data/raw/spokes/
            manifest.csv          # or .tsv / .jsonl

    Expected manifest columns: ``text`` (required), plus any of ``id``,
    ``speaker_id`` (Spokes identifies speakers per conversation -- keep
    them), ``dialect``, ``region``. Samples default to ``family="standard"``;
    dialect columns, when present, override that per row (the derived family
    then wins over the default).
    """

    dataset_name = "spokes"
    acquisition: ClassVar[str] = (
        "manual: export transcripts from https://spokes.clarin-pl.eu/ (CLARIN-PL "
        "account) into data/raw/spokes/manifest.csv with text + speaker_id "
        "columns (see docs/datasets.md)"
    )
    label_defaults: ClassVar[dict[str, str]] = {"family": "standard"}


__all__ = ["SpokesLoader"]
