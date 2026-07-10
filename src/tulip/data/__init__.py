"""Datasets: catalog, loaders, cleaning, deduplication, and leakage-free splits.

Importing this package registers every built-in corpus loader in
:data:`DATASETS`.
"""

from tulip.data import loaders  # noqa: F401  (registration side-effect)
from tulip.data.builder import BUILD_MANIFEST_NAME, DatasetBuilder
from tulip.data.catalog import catalog, get_dataset_info
from tulip.data.cleaning import TextCleaner
from tulip.data.dedup import DedupResult, deduplicate_samples
from tulip.data.download import DownloadReport, DownloadStatus, download_datasets
from tulip.data.manifest import ManifestColumns, read_manifest, surrogate_speaker_id
from tulip.data.reading import read_samples
from tulip.data.registry import DATASETS
from tulip.data.splitting import (
    DatasetSplits,
    load_splits,
    save_splits,
    speaker_disjoint_split,
)

__all__ = [
    "BUILD_MANIFEST_NAME",
    "DATASETS",
    "DatasetBuilder",
    "DatasetSplits",
    "DedupResult",
    "DownloadReport",
    "DownloadStatus",
    "ManifestColumns",
    "TextCleaner",
    "catalog",
    "deduplicate_samples",
    "download_datasets",
    "get_dataset_info",
    "load_splits",
    "read_manifest",
    "read_samples",
    "save_splits",
    "speaker_disjoint_split",
    "surrogate_speaker_id",
]
