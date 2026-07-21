"""A content-addressed store for the Claude dialect baseline's responses.

An API reply is not reproducible from a seed, so the baseline's reproducibility
boundary is this cache: every request is keyed by a digest of what determines its
answer, the first run records each reply, and a later run replays them offline and
byte-identical. An in-memory layer backs an optional on-disk directory: with a
directory, responses survive across processes, which is what lets a second run
replay them; without one, the cache lives only for the process, which suits tests.
Files are written with :func:`tulip._serialize.write_sorted_json`, so a committed
cache stays diff-friendly and regenerable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tulip._serialize import write_sorted_json
from tulip.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["LLMResponseCache"]


class LLMResponseCache:
    """A content-addressed store of model responses, keyed by request digest.

    An in-memory layer backs an optional on-disk directory. With a directory,
    responses survive across processes, which is what lets a second run replay them
    offline; without one, the cache lives only for the process (useful in tests).
    Files are written with :func:`write_sorted_json`, so a committed cache stays
    diff-friendly and regenerable.
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        self._memory: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Return the cached response for ``key``, or ``None`` on a miss.

        A corrupt or partial cache file (invalid JSON, or missing the ``response``
        key) is treated as a miss, so one bad entry re-queries and overwrites itself
        rather than aborting the whole prediction.
        """
        if key in self._memory:
            return self._memory[key]
        if self.directory is not None:
            path = self.directory / f"{key}.json"
            if path.is_file():
                try:
                    response = str(json.loads(path.read_text(encoding="utf-8"))["response"])
                except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
                    logger.debug("ignoring unreadable cache file %s: %s", path, exc)
                    return None
                self._memory[key] = response
                return response
        return None

    def put(self, key: str, response: str, *, request: dict[str, Any] | None = None) -> None:
        """Record ``response`` for ``key`` in memory and (if set) on disk."""
        self._memory[key] = response
        if self.directory is not None:
            payload: dict[str, Any] = {"response": response}
            if request is not None:
                payload["request"] = request
            write_sorted_json(self.directory / f"{key}.json", payload)
