"""Loader for "Dialekty i gwary polskie. Kompendium internetowe" (DGP)."""

from __future__ import annotations

from typing import ClassVar

from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.registry import DATASETS


@DATASETS.register("dgp")
class DgpLoader(ManifestBackedLoader):
    """DGP: curated dialect text samples with fine-grained geography.

    Tier-1 text corpus from the online compendium of Polish dialects
    (https://przewodnik.tmjp.pl/dgp-dialekty-i-gwary-polskie-kompendium-internetowe/).
    Texts are collected manually per the acquisition notes in
    ``docs/datasets.md`` into::

        data/raw/dgp/
            manifest.csv          # or .tsv / .jsonl

    Expected manifest columns: ``text`` (required), plus any of ``id``,
    ``speaker_id``, ``family``, ``dialect``, ``region``, ``village``,
    ``voivodeship``. DGP sample pages usually identify the village and the
    informant; record both when available: village metadata drives the
    surrogate speaker ID when informants are anonymous.
    """

    dataset_name = "dgp"
    acquisition: ClassVar[str] = (
        "manual: no bulk download exists; collect sample texts you are licensed "
        "to use from https://przewodnik.tmjp.pl/dgp-dialekty-i-gwary-polskie-"
        "kompendium-internetowe/ into data/raw/dgp/manifest.csv (see docs/datasets.md)"
    )


__all__ = ["DgpLoader"]
