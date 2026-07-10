"""Loader for Mozilla Common Voice, Polish (https://commonvoice.mozilla.org/)."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.interfaces import DatasetLoader
from tulip.core.types import DatasetInfo, DialectLabels, Sample
from tulip.data.catalog import get_dataset_info
from tulip.data.download import fetch_file
from tulip.data.registry import DATASETS
from tulip.labels.taxonomy import DialectFamily
from tulip.utils.logging import get_logger

_logger = get_logger(__name__)

#: Community mirror hosting the raw CC0 Common Voice release files. The
#: official ``mozilla-foundation`` Hub repo is a script-era dataset with no
#: loadable data files, and Mozilla's portal is email-gated, so this is the
#: only automatable channel; CC0 makes the mirroring licence-clean.
CV_MIRROR_REPO = "fsicoli/common_voice_17_0"

#: Locale fetched by default.
CV_LOCALE = "pl"

#: Columns every Common Voice release TSV is expected to carry.
_REQUIRED_COLUMNS = ("client_id", "path", "sentence")

#: Optional per-speaker metadata columns preserved in ``Sample.metadata``.
_METADATA_COLUMNS = ("age", "gender", "accents", "accent", "variant", "locale", "segment")


@DATASETS.register("common_voice_pl")
class CommonVoiceLoader(DatasetLoader):
    """Mozilla Common Voice (Polish): crowd-read speech with transcripts.

    Uses the official corpus download layout (no manifest assembly needed)::

        data/raw/common_voice_pl/
            validated.tsv         # or train/dev/test.tsv via the tsv param
            clips/<clip>.mp3

    ``client_id`` becomes ``speaker_id`` directly, which is exactly the
    grouping speaker-disjoint splitting needs. The corpus is read standard
    Polish, so samples default to ``family="standard"``; self-reported
    ``accents``/``variant`` values are preserved in ``Sample.metadata`` and
    can be promoted to dialect labels through ``accent_to_dialect`` -- the
    mapping is deliberately explicit because Common Voice accent strings are
    free-form and unreviewed.

    Args:
        tsv: Which release TSV to read (default ``"validated.tsv"``).
        accent_to_dialect: Optional mapping from a (case-insensitive,
            exact-match) accent string to a tulip dialect label; matching
            rows get that dialect instead of the standard-Polish default.
    """

    auto_downloadable: ClassVar[bool] = True

    acquisition: ClassVar[str] = (
        "automatic (text only): `tulip data download common_voice_pl` fetches "
        "validated.tsv from a community mirror of the CC0 release; audio clips "
        "(tens of GB) are not fetched — get them from "
        "https://commonvoice.mozilla.org/en/datasets if you need audio "
        "(see docs/datasets.md)"
    )

    def __init__(
        self,
        tsv: str = "validated.tsv",
        *,
        accent_to_dialect: Mapping[str, str] | None = None,
    ) -> None:
        self._tsv = tsv
        self._accent_to_dialect = {
            key.strip().lower(): value for key, value in (accent_to_dialect or {}).items()
        }

    def download(self, root: Path, **options: Any) -> None:
        """Fetch the release TSV (text + accent metadata; no audio).

        Sources the file from a community mirror of the CC0 release
        (:data:`CV_MIRROR_REPO`): Mozilla's own portal is email/terms-gated
        and the official Hub repo is a script-era dataset that modern
        ``datasets`` cannot load, so the mirror is the only automatable
        channel. CC0 makes mirroring licence-clean; the downloaded header is
        validated so a drifted mirror fails loudly here rather than quietly
        at load time. Audio clips are deliberately not fetched (tens of GB);
        text pipelines work immediately, audio needs the official download.

        Args:
            root: Corpus directory (``data/raw/common_voice_pl``).
            **options: ``limit`` truncates to the first N rows; ``locale``
                (default ``"pl"``), ``repo`` (mirror id), or a full ``url``
                override.

        Raises:
            ConfigurationError: on unknown options.
            DataError: if the transfer fails or the file is not a Common
                Voice release TSV.
        """
        limit = options.pop("limit", None)
        locale = options.pop("locale", CV_LOCALE)
        repo = options.pop("repo", CV_MIRROR_REPO)
        url = options.pop(
            "url",
            f"https://huggingface.co/datasets/{repo}/resolve/main/transcript/{locale}/{self._tsv}",
        )
        if options:
            raise ConfigurationError(
                f"common_voice_pl download got unknown option(s): {', '.join(sorted(options))}"
            )
        destination = root / self._tsv
        fetch_file(url, destination, description=f"Common Voice {locale} {self._tsv}")
        try:
            self._validate_and_truncate(destination, limit)
        except BaseException:
            destination.unlink(missing_ok=True)  # never leave a bad TSV behind
            raise

    def _validate_and_truncate(self, destination: Path, limit: int | None) -> None:
        """Fail loudly on non-CV content; apply the optional row cap."""
        with destination.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
            columns = header.rstrip("\r\n").split("\t")
            missing = [column for column in _REQUIRED_COLUMNS if column not in columns]
            if missing:
                raise DataError(
                    f"downloaded file is not a Common Voice release TSV "
                    f"(missing column(s): {', '.join(missing)}); the mirror may have "
                    "changed layout — pass url=... to point at a good source"
                )
            if limit is None:
                return
            kept = [header, *(line for line, _ in zip(handle, range(limit), strict=False))]
        destination.write_text("".join(kept), encoding="utf-8", newline="")

    @property
    def info(self) -> DatasetInfo:
        """Catalog metadata for Common Voice Polish."""
        return get_dataset_info("common_voice_pl")

    def is_available(self, root: Path) -> bool:
        """Whether the configured release TSV exists under ``root``."""
        return (root / self._tsv).is_file()

    def load(self, root: Path) -> Iterator[Sample]:
        """Yield samples from the official Common Voice TSV layout.

        Raises:
            DataError: if the TSV is missing or lacks the standard columns.
        """
        tsv_path = root / self._tsv
        if not tsv_path.is_file():
            raise DataError(
                f"common_voice_pl: {tsv_path} not found; download the Polish corpus "
                "from https://commonvoice.mozilla.org/ and extract it so the release "
                "TSVs and clips/ sit directly under this directory (see docs/datasets.md)"
            )
        with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = set(reader.fieldnames or ())
            missing = [column for column in _REQUIRED_COLUMNS if column not in fieldnames]
            if missing:
                raise DataError(
                    f"common_voice_pl: {tsv_path} is missing expected column(s) "
                    f"{', '.join(missing)}; is this an official Common Voice release TSV?"
                )
            for line_number, row in enumerate(reader, start=2):
                sample = self._row_to_sample(row, root, tsv_path, line_number)
                if sample is not None:
                    yield sample

    def _row_to_sample(
        self,
        row: Mapping[str, str | None],
        root: Path,
        tsv_path: Path,
        line_number: int,
    ) -> Sample | None:
        """Convert one TSV row to a Sample (``None`` for unusable rows)."""
        sentence = (row.get("sentence") or "").strip()
        clip = (row.get("path") or "").strip()
        speaker = (row.get("client_id") or "").strip()
        if not sentence and not clip:
            _logger.debug("%s:%d: row has neither sentence nor clip", tsv_path, line_number)
            return None
        if not speaker:
            _logger.debug("%s:%d: skipping row without client_id", tsv_path, line_number)
            return None

        accent = (row.get("accents") or row.get("accent") or "").strip()
        dialect = self._accent_to_dialect.get(accent.lower()) if accent else None
        labels = (
            DialectLabels(dialect=dialect)
            if dialect
            else DialectLabels(family=DialectFamily.STANDARD.value)
        )

        metadata = {
            column: value.strip()
            for column in _METADATA_COLUMNS
            if (value := row.get(column)) and value.strip()
        }
        return Sample(
            id=f"common_voice_pl-{Path(clip).stem}" if clip else f"common_voice_pl-{line_number}",
            text=sentence or None,
            audio_path=(root / "clips" / clip) if clip else None,
            speaker_id=speaker,
            labels=labels,
            source="common_voice_pl",
            metadata=metadata,
        )


__all__ = ["CommonVoiceLoader"]
