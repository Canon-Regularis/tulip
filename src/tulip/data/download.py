"""One entry point for acquiring corpora: download what can be downloaded.

Most catalogued corpora have no licence-clean bulk source (see the module
docstring of :mod:`tulip.data.catalog` and ``docs/datasets.md``), so a fully
automatic fetch of everything is not honestly possible. This module does the
next best thing, uniformly:

* corpora whose loader is ``auto_downloadable`` are fetched into the
  documented ``root/<name>/`` layout;
* everything else yields a ``MANUAL`` report carrying the loader's concrete
  acquisition steps, so the user is told exactly what to do instead of
  silently getting nothing.

Adding automatic support for another corpus is one loader change: set
``auto_downloadable = True`` and override ``download`` (see
:class:`tulip.data.loaders.bigos.BigosLoader` for the pattern).
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from tulip.core.exceptions import TulipError
from tulip.data.catalog import catalog
from tulip.data.registry import DATASETS
from tulip.utils.logging import get_logger

_logger = get_logger(__name__)

__all__ = ["DownloadReport", "DownloadStatus", "download_datasets"]


class DownloadStatus(str, enum.Enum):
    """Outcome of one corpus acquisition attempt."""

    DOWNLOADED = "downloaded"
    ALREADY_PRESENT = "already_present"
    MANUAL = "manual"
    FAILED = "failed"


class DownloadReport(BaseModel):
    """What happened for one corpus, in a shape the CLI can render."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: DownloadStatus
    destination: Path
    detail: str


def download_datasets(
    names: Sequence[str] | None,
    root: Path | str,
    *,
    force: bool = False,
    options: dict[str, Any] | None = None,
) -> list[DownloadReport]:
    """Acquire the requested corpora under ``root``, one report per corpus.

    Args:
        names: Canonical corpus names, or ``None`` for every catalogued
            corpus (tier order).
        root: Local corpora root (each corpus lands in ``root/<name>/``).
        force: Re-download corpora that already look present locally.
        options: Downloader knobs forwarded to every auto-downloadable
            loader's ``download`` (e.g. ``{"limit": 10_000}``); loaders
            reject options they do not understand.

    Returns:
        One :class:`DownloadReport` per requested corpus, in request order.
        ``MANUAL`` reports carry the acquisition steps in ``detail``; a
        failing automatic download yields a ``FAILED`` report (with the error
        and any remediation steps in ``detail``) rather than aborting the
        remaining corpora — one gated dataset must not sink an ``--all`` run.

    Raises:
        UnknownComponentError: if a name is not a registered dataset (a
            caller error, unlike a per-corpus download failure).
    """
    root = Path(root)
    requested = list(names) if names is not None else [info.name for info in catalog()]
    reports: list[DownloadReport] = []
    for name in requested:
        loader = DATASETS.get(name)()  # default construction; download() takes the knobs
        destination = root / loader.info.name
        if not force and loader.is_available(destination):
            reports.append(
                DownloadReport(
                    name=loader.info.name,
                    status=DownloadStatus.ALREADY_PRESENT,
                    destination=destination,
                    detail="local copy found (use --force to re-download)",
                )
            )
            continue
        if loader.auto_downloadable:
            _logger.info("downloading %s -> %s", loader.info.name, destination)
            try:
                loader.download(destination, **(options or {}))
            except TulipError as exc:
                _logger.warning("download of %s failed: %s", loader.info.name, exc)
                reports.append(
                    DownloadReport(
                        name=loader.info.name,
                        status=DownloadStatus.FAILED,
                        destination=destination,
                        detail=str(exc),
                    )
                )
                continue
            reports.append(
                DownloadReport(
                    name=loader.info.name,
                    status=DownloadStatus.DOWNLOADED,
                    destination=destination,
                    detail=f"fetched into {destination}",
                )
            )
            continue
        reports.append(
            DownloadReport(
                name=loader.info.name,
                status=DownloadStatus.MANUAL,
                destination=destination,
                detail=loader.acquisition
                or f"no automatic download; see {loader.info.url} and docs/datasets.md",
            )
        )
    return reports
