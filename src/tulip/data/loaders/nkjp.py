"""Loader for the National Corpus of Polish (NKJP) as standard-Polish negatives."""

from __future__ import annotations

from typing import ClassVar

from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.manifest import ManifestColumns
from tulip.data.registry import DATASETS


@DATASETS.register("nkjp")
class NkjpLoader(ManifestBackedLoader):
    """NKJP (https://nkjp.pl/): standard-Polish negative examples.

    Tier-3 corpus used as the *negative* class for dialect-vs-standard
    tasks: every sample is labelled ``family="standard"`` and dialect-level
    manifest columns are deliberately ignored (NKJP text is general Polish
    regardless of the author's origin). Export sentences/paragraphs locally
    (see ``docs/datasets.md``) into::

        data/raw/nkjp/
            manifest.csv          # or .tsv / .jsonl

    Expected manifest columns: ``text`` (required), plus any of ``id``,
    ``speaker_id`` (use the NKJP document/author identifier so all text from
    one source stays in one split). Rows without a ``speaker_id`` fall back
    to one surrogate speaker per manifest file.
    """

    dataset_name = "nkjp"
    acquisition: ClassVar[str] = (
        "manual: download the NKJP-1M balanced subcorpus from "
        "http://clip.ipipan.waw.pl/NationalCorpusOfPolish (GNU GPL), extract "
        "paragraph text into data/raw/nkjp/manifest.csv with a `text` column "
        "(see docs/datasets.md)"
    )
    columns: ClassVar[ManifestColumns] = ManifestColumns(
        family=None, dialect=None, region=None, village=None, voivodeship=None
    )
    label_defaults: ClassVar[dict[str, str]] = {"family": "standard"}


__all__ = ["NkjpLoader"]
