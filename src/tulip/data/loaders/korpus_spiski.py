"""Loader for the Spisz dialect corpus (Korpus Spiski)."""

from __future__ import annotations

from typing import ClassVar

from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.registry import DATASETS


@DATASETS.register("korpus_spiski")
class KorpusSpiskiLoader(ManifestBackedLoader):
    """Korpus Spiski: transcribed speech from the Spisz region.

    Tier-2 single-dialect corpus (see
    https://journals.akademicka.pl/lv/article/view/727). Assemble locally
    into::

        data/raw/korpus_spiski/
            manifest.csv          # or .tsv / .jsonl

    Expected manifest columns: ``text`` (required), plus any of ``id``,
    ``speaker_id``, ``village``, ``region``. Every sample defaults to
    ``dialect="spisz"`` (family ``lesser_polish`` derives automatically);
    a manifest ``dialect`` column overrides the default if present.
    """

    dataset_name = "korpus_spiski"
    label_defaults: ClassVar[dict[str, str]] = {"dialect": "spisz"}


__all__ = ["KorpusSpiskiLoader"]
