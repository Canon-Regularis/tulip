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
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from tulip.core.exceptions import DataError, TulipError
from tulip.data.catalog import catalog
from tulip.data.registry import DATASETS
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = get_logger(__name__)

#: URL schemes :func:`fetch_file` will open. ``file:`` is kept deliberately:
#: offline mirrors and the test fixtures rely on it, but everything outside
#: this set (``ftp:``, ``data:``, and any custom-registered urllib opener) is
#: refused. Passing an arbitrary scheme straight to ``urlopen`` is what makes
#: the call surprising; an explicit allow-list makes the supported set the
#: contract, and turns a typo'd catalog URL into a clear error.
_ALLOWED_URL_SCHEMES = frozenset({"http", "https", "file"})

__all__ = ["DownloadReport", "DownloadStatus", "download_datasets", "fetch_file"]


def _validate_url_scheme(url: str, *, description: str) -> None:
    """Reject URL schemes outside :data:`_ALLOWED_URL_SCHEMES`.

    Raises:
        DataError: if the scheme is missing or not allowed.
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        allowed = ", ".join(sorted(_ALLOWED_URL_SCHEMES))
        raise DataError(
            f"refusing to fetch {description}: URL scheme {scheme or '(none)'!r} is not "
            f"allowed (permitted schemes: {allowed}); offending URL: {url}"
        )


def fetch_file(url: str, destination: Path, *, description: str) -> Path:
    """Stream a URL to ``destination``, never leaving a partial file behind.

    Shared by the corpus downloaders: writes to ``<destination>.part`` and
    renames only on success, so an interrupted or failed transfer cannot
    masquerade as an acquired corpus. ``file://`` URLs work too (used by
    tests and offline mirrors).

    Args:
        url: Source URL (redirects are followed).
        destination: Final file path; parent directories are created.
        description: Human label for logs and error messages.

    Returns:
        ``destination``.

    Raises:
        DataError: if the URL scheme is not allowed, or the transfer fails for
            any reason.
    """
    # Validated before any filesystem work, and outside the try block below, so
    # a rejected scheme surfaces as itself rather than as "could not download".
    _validate_url_scheme(url, description=description)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    _logger.info("downloading %s from %s", description, url)
    try:
        # HTTP(S) redirects and file:// are both handled by urllib openers.
        # S310 is suppressed below because _validate_url_scheme has already
        # restricted `url` to the allow-listed schemes.
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:  # noqa: S310
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        partial.replace(destination)
    except BaseException as exc:
        partial.unlink(missing_ok=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise DataError(f"could not download {description} from {url}: {exc}") from exc
    _logger.info("%s downloaded (%.1f MB)", description, destination.stat().st_size / (1024 * 1024))
    return destination


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


def _options_for(loader: object, options: dict[str, Any] | None) -> dict[str, Any]:
    """Per-loader download options, dropping ``audio`` where it is not supported.

    ``audio`` is a batch-wide flag: on ``--all`` it reaches every loader, but
    only some (bigos, common_voice_pl) can fetch audio. Rather than fail the
    rest on an unknown option, drop ``audio`` for loaders that do not advertise
    ``supports_audio_fetch``, so those corpora still acquire their text. A per
    call copy keeps one loader's pop from touching the next.
    """
    loader_options = dict(options or {})
    if "audio" in loader_options and not getattr(loader, "supports_audio_fetch", False):
        loader_options.pop("audio")
    return loader_options


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
        remaining corpora. One gated dataset must not sink an ``--all`` run.

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
            loader_options = _options_for(loader, options)
            try:
                loader.download(destination, **loader_options)
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
