"""Atomic streamed-manifest writing for the Hub dataset loaders.

The BIGOS and Common Voice loaders both stream records from the Hugging Face Hub
into a local CSV/TSV manifest, writing each record's clip as they go. They share
one discipline: on any failure mid-stream (a gated dataset, a dropped connection,
Ctrl+C) the partial manifest and every clip written this run must be removed, so a
broken fetch never masquerades as a present corpus, and a stream that yields
nothing is an error. This holds that discipline in one place; each loader supplies
only its header and its per-record row builder.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from tulip.core.exceptions import DataError, TulipError
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path
    from typing import Any

__all__ = ["stream_records_to_manifest"]

_logger = get_logger(__name__)


def stream_records_to_manifest(
    output_path: Path,
    records: Iterable[Any],
    row_builder: Callable[[Any, int], tuple[Sequence[Any], Sequence[Path]] | None],
    *,
    header: Sequence[Any],
    limit: int | None,
    source: str,
    empty_error: str,
    delimiter: str = ",",
    progress_every: int | None = None,
) -> int:
    """Stream ``records`` into a CSV/TSV manifest, cleaning up on any failure.

    Writes the header, pulls records until ``limit``, and delegates each to
    ``row_builder``, which returns its manifest row and the clip files it wrote
    (or ``None`` to skip the record). The returned clips are tracked so that, on
    any failure, the partial manifest and every clip written this run are removed.
    A control-flow exception (``KeyboardInterrupt`` / ``SystemExit``) or a
    :class:`~tulip.core.exceptions.TulipError` propagates unchanged; anything else
    becomes a mid-stream :class:`~tulip.core.exceptions.DataError`. A stream that
    yields no usable record is an error too.

    Args:
        output_path: Manifest file to write.
        records: The stream of raw records.
        row_builder: ``(record, index) -> (row, clips)``, or ``None`` to skip. The
            returned clips are tracked for cleanup on failure.
        header: The manifest header row.
        limit: Maximum rows to write, or ``None`` for all.
        source: Loader name used in the mid-stream error message and progress log.
        empty_error: Message raised when the stream yields no usable record.
        delimiter: Field delimiter; pass ``"\\t"`` for a TSV.
        progress_every: Log a progress line every N rows, or ``None`` for quiet.

    Returns:
        The number of rows written.

    Raises:
        DataError: on a mid-stream failure, or when nothing usable is produced.
    """
    count = 0
    written_clips: list[Path] = []
    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(list(header))
            for index, record in enumerate(records):
                if limit is not None and count >= limit:
                    break
                built = row_builder(record, index)
                if built is None:
                    continue
                row, clips = built
                written_clips.extend(clips)
                writer.writerow(list(row))
                count += 1
                if progress_every is not None and count % progress_every == 0:
                    _logger.info("%s download: %d samples written", source, count)
    except BaseException as exc:
        # Never leave a partial manifest (it would masquerade as a present corpus)
        # nor clips this run wrote (they would leak on retry).
        output_path.unlink(missing_ok=True)
        for clip_path in written_clips:
            clip_path.unlink(missing_ok=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit, TulipError)):
            raise
        raise DataError(f"{source} download failed mid-stream: {exc}") from exc
    if count == 0:
        output_path.unlink(missing_ok=True)
        raise DataError(empty_error)
    return count
