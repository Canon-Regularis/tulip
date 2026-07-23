"""Output-directory preparation shared by the corpus writers.

Every writer that lays down a manifest (synthetic text, synthetic audio,
transcripts) and the transcript cache all begin the same way: make the target
directory, parents included, tolerating one that is already there. The failure
they all share is a caller pointing ``--out`` or ``--cache`` at a path that is
already a regular file, which surfaces from ``mkdir`` as a bare ``FileExistsError``
(or ``NotADirectoryError`` further down the path). One helper turns that into the
same clean, actionable error everywhere, so no writer has to restate it.
"""

from __future__ import annotations

from pathlib import Path

from tulip.core.exceptions import DataError

__all__ = ["ensure_directory"]


def ensure_directory(path: Path | str, *, purpose: str) -> Path:
    """Create ``path`` as a directory, parents included, and return it.

    Args:
        path: The directory to create; an existing directory is left alone.
        purpose: What the directory is for, quoted in the error (e.g.
            ``"synthetic manifest"``), so a caller reads which option was wrong.

    Returns:
        The directory as a :class:`~pathlib.Path`.

    Raises:
        DataError: if the path exists as a file, or cannot be created.
    """
    directory = Path(path)
    if directory.exists() and not directory.is_dir():
        raise DataError(
            f"the {purpose} output path {directory} is an existing file, not a directory"
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataError(
            f"could not create the {purpose} output directory {directory}: {exc}"
        ) from exc
    return directory
