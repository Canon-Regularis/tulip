"""Loader for Mozilla Common Voice, Polish (https://commonvoice.mozilla.org/)."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.interfaces import DatasetLoader
from tulip.core.types import DatasetInfo, DialectLabels, Sample
from tulip.data.catalog import get_dataset_info
from tulip.data.download import fetch_file
from tulip.data.loaders._hub_audio import CLIPS_DIR, write_hub_clip
from tulip.data.loaders._stream import stream_records_to_manifest
from tulip.data.registry import DATASETS
from tulip.labels.taxonomy import DialectFamily
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    can be promoted to dialect labels through ``accent_to_dialect``; the
    mapping is deliberately explicit because Common Voice accent strings are
    free-form and unreviewed.

    Args:
        tsv: Which release TSV to read (default ``"validated.tsv"``).
        accent_to_dialect: Optional mapping from a (case-insensitive,
            exact-match) accent string to a tulip dialect label; matching
            rows get that dialect instead of the standard-Polish default.
    """

    auto_downloadable: ClassVar[bool] = True

    #: ``download(audio=True)`` streams and materialises the clips (see the
    #: batch orchestrator in :func:`tulip.data.download.download_datasets`).
    supports_audio_fetch: ClassVar[bool] = True

    acquisition: ClassVar[str] = (
        "automatic (text only by default): `tulip data download common_voice_pl` "
        "fetches validated.tsv from a community mirror of the CC0 release; add "
        "`--audio` (with `--limit`) to also stream and materialise the clips "
        "(needs the `hf` extra; the full corpus is tens of GB) (see docs/datasets.md)"
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
        """Fetch the release TSV (text + accent metadata), optionally with audio.

        Text mode (default) sources the release TSV from a community mirror of
        the CC0 release (:data:`CV_MIRROR_REPO`): Mozilla's own portal is
        email/terms-gated and the official Hub repo is a script-era dataset
        that modern ``datasets`` cannot load, so the mirror is the only
        automatable channel. CC0 makes mirroring licence-clean; the downloaded
        header is validated so a drifted mirror fails loudly here rather than
        quietly at load time.

        Audio mode (``audio=True``, the CLI's ``--audio``) instead streams the
        clips from the same mirror with the ``datasets`` library, writes them
        under ``clips/`` and a matching release TSV alongside, so the standard
        :meth:`load` picks up both text and audio offline. This is the
        first-party real audio corpus the speech models need; pair it with
        ``limit`` because the full corpus is tens of GB.

        Args:
            root: Corpus directory (``data/raw/common_voice_pl``).
            **options: ``limit`` caps the number of rows/clips; ``locale``
                (default ``"pl"``) and ``repo`` (mirror id) apply to both
                modes; ``url`` overrides the TSV source (text mode only);
                ``audio`` fetches the clips and ``split`` (default ``"train"``)
                picks the streamed split (audio mode only).

        Raises:
            ConfigurationError: on unknown options.
            DataError: if the transfer fails, the file is not a Common Voice
                release TSV, or the audio stream yields no samples.
            MissingDependencyError: in audio mode without the ``hf`` extra.
        """
        limit = options.pop("limit", None)
        locale = options.pop("locale", CV_LOCALE)
        repo = options.pop("repo", CV_MIRROR_REPO)
        if options.pop("audio", False):
            split = options.pop("split", "train")
            if options:
                raise ConfigurationError(
                    f"common_voice_pl audio download got unknown option(s): "
                    f"{', '.join(sorted(options))}"
                )
            self._download_audio(root, repo=repo, locale=locale, split=split, limit=limit)
            return
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

    def _download_audio(
        self, root: Path, *, repo: str, locale: str, split: str, limit: int | None
    ) -> None:
        """Stream clips + transcripts from the mirror into the official layout.

        Writes each streamed record's clip under ``clips/`` and a release TSV
        (``self._tsv``) carrying the same ``path`` values, so :meth:`load`
        reads the materialised corpus with no special-casing. The TSV is
        removed on any failure so a partial fetch never masquerades as a
        present corpus.
        """
        datasets = optional_import(
            "datasets", extra="hf", purpose="streaming Common Voice audio from the Hugging Face Hub"
        )
        _logger.info(
            "streaming %s (locale=%s, split=%s) audio from the Hugging Face Hub",
            repo,
            locale,
            split,
        )
        try:
            stream = datasets.load_dataset(repo, locale, split=split, streaming=True)
            # Disable decoding so each record carries the clip's original encoded
            # bytes: audio fetch writes them verbatim and needs no decode backend.
            stream = stream.cast_column("audio", datasets.Audio(decode=False))
        except Exception as exc:  # datasets raises many library-specific types
            raise DataError(
                f"common_voice_pl: could not stream {repo!r} (locale={locale!r}, "
                f"split={split!r}) audio from the Hub: {exc}"
            ) from exc
        root.mkdir(parents=True, exist_ok=True)
        clips_dir = root / CLIPS_DIR
        tsv_path = root / self._tsv

        def build_row(record: Mapping[str, Any], index: int) -> tuple[list[str], list[Path]] | None:
            row = self._audio_record_row(record, clips_dir, index)
            if row is None:
                return None
            return row, [clips_dir / row[1]]

        count = stream_records_to_manifest(
            tsv_path,
            stream,
            build_row,
            header=["client_id", "path", "sentence", "accents"],
            limit=limit,
            source="common_voice_pl",
            empty_error=(
                f"common_voice_pl audio download produced no samples; check the mirror "
                f"({repo!r}, locale={locale!r}, split={split!r})"
            ),
            delimiter="\t",
        )
        _logger.info("common_voice_pl audio download complete: %d clips -> %s", count, clips_dir)

    def _audio_record_row(
        self, record: Mapping[str, Any], clips_dir: Path, index: int
    ) -> list[str] | None:
        """Materialise one streamed record's clip; return its release-TSV row.

        ``None`` for rows without a ``client_id`` (speaker-disjoint splitting
        needs one). The clip is named from the record's original ``path``,
        prefixed with the stream index so two identical stems cannot collide,
        and the same name goes into the TSV so it and the clip stay in step.
        """
        speaker = str(record.get("client_id") or "").strip()
        if not speaker:
            return None
        audio = record.get("audio")
        original = str(record.get("path") or "").strip()
        if not original and isinstance(audio, Mapping):
            original = str(audio.get("path") or "").strip()
        stem = f"{index:08d}-{Path(original).stem or 'clip'}"
        name = write_hub_clip(audio, clips_dir, stem)
        sentence = str(record.get("sentence") or "").strip()
        accent = str(record.get("accents") or record.get("accent") or "").strip()
        return [speaker, name, sentence, accent]

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
                    "changed layout; pass url=... to point at a good source"
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
