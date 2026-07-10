"""Loader for the Mackowce dialect text corpus (EKTGM)."""

from __future__ import annotations

from typing import ClassVar

from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.registry import DATASETS


@DATASETS.register("mackowce")
class MackowceLoader(ManifestBackedLoader):
    """Elektroniczny Korpus Tekstow Gwarowych z Mackowiec (Podolia).

    Tier-2 single-locality corpus of borderland Polish from Mackowce (see
    https://przewodnik.tmjp.pl/ektgm-elektroniczny-korpus-tekstow-gwarowych-z-mackowiec-na-podolu/).
    Assemble locally into::

        data/raw/mackowce/
            manifest.csv          # or .tsv / .jsonl

    Expected manifest columns: ``text`` (required), plus any of ``id``,
    ``speaker_id``. Every sample defaults to ``dialect="podolia"`` (the
    taxonomy's label for the Mackowce borderland variety) and
    ``village="Mackowce"``; manifest columns override the defaults.
    """

    dataset_name = "mackowce"
    acquisition: ClassVar[str] = (
        "manual: no bulk download exists; collect texts from the EKTGM corpus "
        "(https://przewodnik.tmjp.pl/ektgm-elektroniczny-korpus-tekstow-gwarowych"
        "-z-mackowiec-na-podolu/) into data/raw/mackowce/manifest.csv "
        "(see docs/datasets.md)"
    )
    label_defaults: ClassVar[dict[str, str]] = {
        "dialect": "podolia",
        "village": "Mackowce",
    }


__all__ = ["MackowceLoader"]
