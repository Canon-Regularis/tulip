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
from tulip.data.synthetic import SyntheticSpec, generate_corpus, write_synthetic_manifest
from tulip.data.validation import ManifestIssue, ManifestReport, validate_manifest

__all__ = [
    "BUILD_MANIFEST_NAME",
    "DATASETS",
    "DatasetBuilder",
    "DatasetSplits",
    "DedupResult",
    "DownloadReport",
    "DownloadStatus",
    "ManifestColumns",
    "ManifestIssue",
    "ManifestReport",
    "SyntheticSpec",
    "TextCleaner",
    "catalog",
    "deduplicate_samples",
    "download_datasets",
    "generate_corpus",
    "get_dataset_info",
    "load_splits",
    "read_manifest",
    "read_samples",
    "save_splits",
    "speaker_disjoint_split",
    "surrogate_speaker_id",
    "validate_manifest",
    "write_synthetic_manifest",
]
