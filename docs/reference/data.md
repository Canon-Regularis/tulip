# Data

The dataset subsystem: the catalog, the `DatasetBuilder` that orchestrates
load → clean → dedup → speaker-disjoint split → persist, manifest and validation
helpers, and the synthetic-corpus generator. See [Datasets](../datasets.md) for
acquisition and the manifest format.

## Building splits

::: tulip.data.DatasetBuilder

::: tulip.data.speaker_disjoint_split

::: tulip.data.DatasetSplits

::: tulip.data.save_splits

::: tulip.data.load_splits

## Catalog and acquisition

::: tulip.data.catalog

::: tulip.data.get_dataset_info

::: tulip.data.download_datasets

## Reading, cleaning, and deduplication

::: tulip.data.read_samples

::: tulip.data.read_manifest

::: tulip.data.TextCleaner

::: tulip.data.deduplicate_samples

## Validation

::: tulip.data.validate_manifest

## Reproducibility

A content fingerprint over a produced split, so its byte-for-byte reproducibility
is verifiable — not just its sizes. `DatasetBuilder.build` writes one as
`split_lock.json` next to the splits.

::: tulip.data.fingerprint_splits

::: tulip.data.SplitFingerprint

::: tulip.data.verify_splits

## Synthetic corpus

::: tulip.data.generate_corpus

::: tulip.data.SyntheticSpec

::: tulip.data.write_synthetic_manifest
