"""Reading labelled samples back from files tulip (or a researcher) wrote.

Complements :mod:`tulip.data.manifest` (which ingests *foreign* corpora):
this module reads sample collections in any of the shapes tulip itself
produces or documents -- split JSONL files written by
:func:`tulip.data.splitting.save_splits`, manifest files, or a directory
containing a manifest. The CLI and evaluation entry points share it so
"anything labelled" is accepted uniformly everywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from tulip.core.exceptions import DataError
from tulip.core.types import Sample
from tulip.data.manifest import read_manifest
from tulip.utils.io import read_jsonl
from tulip.utils.logging import get_logger

_logger = get_logger(__name__)

__all__ = ["read_samples"]

_SPLIT_FILE_SUFFIXES = {".jsonl", ".ndjson"}


def read_samples(path: Path | str) -> Iterator[Sample]:
    """Yield labelled samples from a split file, manifest file, or directory.

    Accepted shapes, tried in order:

    * a directory -- probed for ``manifest.{csv|tsv|jsonl}``;
    * a ``.jsonl``/``.ndjson`` file of serialised :class:`Sample` records
      (as written by :func:`~tulip.data.splitting.save_splits`), falling back
      to the manifest row format when the records do not validate;
    * any other file -- parsed as a manifest (CSV/TSV).

    Raises:
        DataError: if the path does not exist or no interpretation succeeds.
    """
    path = Path(path)
    if path.is_dir():
        from tulip.data.registry import DATASETS

        yield from DATASETS.create("manifest").load(path)
        return
    if not path.is_file():
        raise DataError(f"no such file or directory: {path}")
    if path.suffix.lower() in _SPLIT_FILE_SUFFIXES:
        # Parse fully before yielding: a mid-stream validation failure must
        # fall back to the manifest format without emitting partial results.
        try:
            samples = [Sample.model_validate(record) for record in read_jsonl(path)]
        except (ValueError, TypeError) as exc:  # not split-file records
            _logger.debug("%s is not a split file (%s); trying manifest format", path, exc)
        else:
            yield from samples
            return
    yield from read_manifest(path)
