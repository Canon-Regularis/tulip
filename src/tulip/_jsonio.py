"""Read a JSON artifact that must contain an object.

Model sidecars, the fusion and hierarchical sidecars, the deploy registry index,
and the card commands all read a small JSON file that must parse to an object,
and each wrapped the same two failures (unreadable, or parsed to something that
is not an object) in a :class:`~tulip.core.exceptions.DataError`. This
centralises that reader so a corrupt artifact always fails with one clean error
instead of a raw traceback; a few sites previously omitted the guard and leaked a
``JSONDecodeError``.

It lives at the package root, beside :mod:`tulip._serialize`, so every layer
(models, pipeline, deploy, cli) can share it without a dependency cycle; it
imports only from ``core`` and the frozen ``utils`` reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.core.exceptions import DataError
from tulip.utils.io import read_json

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

__all__ = ["read_json_object"]


def read_json_object(path: Path, *, what: str) -> dict[str, Any]:
    """Read ``path`` as JSON that must be an object, else raise :class:`DataError`.

    Args:
        path: The JSON file to read.
        what: A short noun for the artifact, e.g. ``"metadata sidecar"``; it
            names the file in both error messages.

    Returns:
        The parsed JSON object as a dict.

    Raises:
        DataError: if the file is missing or unparseable, or if it parses to
            something other than a JSON object.
    """
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:  # JSONDecodeError subclasses ValueError
        raise DataError(f"{what} at {path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataError(f"{what} at {path} must be a JSON object, got {type(payload).__name__}")
    return payload
