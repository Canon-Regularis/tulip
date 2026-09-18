"""Loader for BIGOS (https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2)."""

from __future__ import annotations

import csv
import tarfile
from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import Sample
from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.loaders._hub_audio import CLIPS_DIR
from tulip.data.loaders._stream import stream_records_to_manifest
from tulip.data.manifest import surrogate_speaker_id
from tulip.data.registry import DATASETS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

_logger = get_logger(__name__)

#: Field names probed for a speaker identifier in Hub records.
_SPEAKER_FIELDS = ("speaker_id", "speakerid", "speaker", "client_id")

#: Field names probed for the transcription text in Hub records.
_TEXT_FIELDS = ("ref_orig", "ref", "transcription", "sentence", "text")

#: The maintained BIGOS release. The original ``michaljunczyk/pl-asr-bigos`` is
#: frozen at 1,900 recordings and ships a ``test`` split only, so its records
#: cannot be reached through this loader's ``train`` default at all.
DEFAULT_HF_DATASET = "amu-cai/pl-asr-bigos-v2"

#: BIGOS is published as a loading script, and ``datasets`` 4.0 removed script
#: support, so ``load_dataset`` cannot read the dataset branch on any modern
#: install. The Hub's automatic parquet conversion of the same data is loadable,
#: so that is what this loader streams.
PARQUET_REVISION = "refs/convert/parquet"

#: Per-subset audio lives in release tarballs; the parquet conversion drops the
#: audio column, so clips are matched to records on this field.
_AUDIO_NAME_FIELD = "audiopath_bigos"


@DATASETS.register("bigos")
class BigosLoader(ManifestBackedLoader):
    """BIGOS: aggregated Polish ASR benchmark (audio + transcriptions).

    Two acquisition modes:

    * **Local manifest** (default): assemble ``data/raw/bigos/manifest.csv``
      like any other corpus (see ``docs/datasets.md``).
    * **Hugging Face Hub** (``from_hub=True``): stream transcriptions
      directly with the ``datasets`` library (extra ``hf``). Hub mode yields
      *text-only* samples; to also materialise the audio use
      ``tulip data download bigos --audio`` (see :meth:`download`), which
      writes the clips locally and records their paths so the speech models
      have a first-party real corpus.

    Dialect labels are not provided (tier 4): BIGOS is pretraining/ASR
    material, so labels stay empty unless your manifest adds them.

    Args:
        manifest: Manifest name for local mode (default probe order).
        from_hub: Load transcriptions from the Hugging Face Hub instead of a
            local manifest.
        hf_dataset: Hub dataset id for hub mode (default
            :data:`DEFAULT_HF_DATASET`).
        hf_config: Optional Hub configuration (source-corpus subset) name.
        revision: Hub revision to stream. Defaults to
            :data:`PARQUET_REVISION` because the dataset branch is a loading
            script, which ``datasets>=4`` refuses to execute.
        split: Hub split to stream (default ``"train"``).
        subsets: Restrict to these source corpora (e.g. ``("pwr-viu-unk",)``).
            The records stream in subset order, so this is what makes a
            partial audio fetch representative rather than just the first
            subset, and it bounds how many tarballs an audio fetch pulls.
        limit: Optional cap on the number of hub samples (useful for smoke
            experiments; ``None`` streams everything).
    """

    dataset_name: ClassVar[str] = "bigos"

    auto_downloadable: ClassVar[bool] = True

    #: ``download(audio=True)`` streams and materialises the clips (see the
    #: batch orchestrator in :func:`tulip.data.download.download_datasets`).
    supports_audio_fetch: ClassVar[bool] = True

    acquisition: ClassVar[str] = (
        "automatic with a Hugging Face login (the dataset is gated): accept the "
        f"conditions at https://huggingface.co/datasets/{DEFAULT_HF_DATASET}, "
        "authenticate (`hf auth login` or HF_TOKEN), then `tulip data download "
        "bigos` streams the transcriptions into data/raw/bigos/manifest.csv "
        "(text-only; add --audio for the clips; requires the `hf` extra)"
    )

    def __init__(
        self,
        manifest: str | None = None,
        *,
        from_hub: bool = False,
        hf_dataset: str = DEFAULT_HF_DATASET,
        hf_config: str | None = None,
        revision: str | None = PARQUET_REVISION,
        split: str = "train",
        subsets: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(manifest)
        self._from_hub = from_hub
        self._hf_dataset = hf_dataset
        self._hf_config = hf_config
        self._revision = revision
        self._split = split
        self._subsets = frozenset(subsets) if subsets else None
        self._limit = limit

    def is_available(self, root: Path) -> bool:
        """Hub mode is always 'available'; local mode needs a manifest."""
        if self._from_hub:
            return True
        return super().is_available(root)

    def load(self, root: Path) -> Iterator[Sample]:
        """Yield samples from the local manifest or the Hugging Face Hub."""
        if not self._from_hub:
            yield from super().load(root)
            return
        yield from self._load_from_hub(limit=self._limit)

    def download(self, root: Path, **options: Any) -> None:
        """Materialise the Hub transcriptions as a local ``manifest.csv``.

        After this, the default (manifest) mode works fully offline. By default
        audio is not fetched (BIGOS audio is tens of GB); pass ``audio=True``
        (the CLI's ``--audio``) to also write each sample's clip under
        ``clips/`` and record its relative path in the manifest, which is what
        gives the speech models a first-party real corpus. Combine with
        ``limit`` to fetch a tractable slice.

        Args:
            root: Corpus directory (``data/raw/bigos``).
            **options: ``limit`` overrides the constructor's sample cap;
                ``audio`` also fetches the clips (verbatim from the release
                tarballs, so no audio extra is needed).

        Raises:
            ConfigurationError: on unknown options.
            DataError: if the Hub yields no samples, or a tarball cannot be
                fetched in audio mode.
            MissingDependencyError: without the ``hf`` extra installed.
        """
        limit = options.pop("limit", self._limit)
        audio = bool(options.pop("audio", False))
        if options:
            raise ConfigurationError(
                f"bigos download got unknown option(s): {', '.join(sorted(options))}"
            )
        root.mkdir(parents=True, exist_ok=True)
        if audio:
            self._download_with_audio(root, limit=limit)
            return

        manifest_path = root / "manifest.csv"
        header = ["id", "text", "speaker_id", "subset"]
        retained = self._rows_to_retain(manifest_path, header)

        def build_row(record: dict[str, Any], index: int) -> tuple[list[Any], list[Path]] | None:
            sample = self._record_to_sample(record, index)
            if sample is None:
                return None
            row: list[Any] = [
                sample.id,
                sample.text,
                sample.speaker_id,
                str(sample.metadata.get("dataset", "")),
            ]
            return row, []

        count = stream_records_to_manifest(
            manifest_path,
            self._stream_records(),
            build_row,
            header=header,
            limit=limit,
            source="bigos",
            empty_error=(
                "bigos download produced no samples; check the Hub dataset "
                f"({self._hf_dataset!r}, config={self._hf_config!r}, split={self._split!r})"
            ),
            progress_every=5000,
        )
        self._append_retained(manifest_path, retained)
        _logger.info("bigos download complete: %d samples -> %s", count, manifest_path)

    def _rows_to_retain(self, manifest_path: Path, header: list[str]) -> list[list[str]]:
        """Existing manifest rows this fetch must not delete.

        Audio ships as one tarball per source corpus, so a corpus is naturally
        assembled a few subsets at a time. Rewriting the whole manifest on every
        fetch would make each run silently discard the last one's work. A
        subset-scoped fetch therefore keeps the rows belonging to subsets it was
        not asked for; an unscoped fetch still replaces everything, which is what
        ``--force`` and a full re-fetch should do.

        Rows are kept only when the existing file has exactly this fetch's
        columns, so a text-only manifest and an audio one never merge into a
        ragged file.
        """
        if not self._subsets or not manifest_path.is_file():
            return []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            existing = next(reader, None)
            if existing != header:
                _logger.info(
                    "replacing %s rather than merging: its columns %s differ from this fetch's %s",
                    manifest_path.name,
                    existing,
                    header,
                )
                return []
            subset_at = header.index("subset")
            return [row for row in reader if row[subset_at] not in self._subsets]

    def _append_retained(self, manifest_path: Path, retained: list[list[str]]) -> None:
        """Re-attach the rows :meth:`_rows_to_retain` set aside."""
        if not retained:
            return
        with manifest_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(retained)
        _logger.info("kept %d existing row(s) from subsets outside this fetch", len(retained))

    def _download_with_audio(self, root: Path, *, limit: int | None) -> None:
        """Write the manifest and materialise its clips from the release tarballs.

        The parquet conversion this loader streams carries transcriptions and
        speaker ids but no audio, and the dataset branch that does carry audio is
        unreadable on ``datasets>=4``. The per-subset ``.tar.gz`` releases are
        plain files, so the clips come from those, matched to records on
        :data:`_AUDIO_NAME_FIELD`.

        Records are collected first so each tarball is opened once and scanned
        once, and only the wanted members are extracted: the full release is
        about 28 GB, and a limited fetch should not pay for it.
        """
        hub = optional_import(
            "huggingface_hub", extra="hf", purpose="fetching BIGOS audio from the Hub"
        )

        wanted: dict[str, dict[str, Sample]] = defaultdict(dict)
        ordered: list[tuple[Sample, str, str]] = []
        seen = 0
        for index, record in enumerate(self._stream_records()):
            if limit is not None and len(ordered) >= limit:
                break
            seen += 1
            sample = self._record_to_sample(record, index)
            if sample is None:
                continue
            clip_name = str(record.get(_AUDIO_NAME_FIELD) or "").strip()
            if not clip_name:
                continue
            subset = str(record.get("dataset") or "")
            wanted[subset][clip_name] = sample
            ordered.append((sample, subset, clip_name))

        if not ordered:
            detail = (
                f"{seen} record(s) matched but none carried a transcription; BIGOS "
                "withholds references on its benchmark test splits, so use "
                "split='train' (or 'validation') for supervised material"
                if seen
                else "no records matched"
            )
            raise DataError(
                f"bigos audio download produced no samples ({self._hf_dataset!r}, "
                f"split={self._split!r}, subsets={self._subsets}): {detail}"
            )

        clips_dir = root / CLIPS_DIR
        clips_dir.mkdir(parents=True, exist_ok=True)
        extracted: set[str] = set()
        for subset, names in wanted.items():
            archive = f"data/{subset}/{self._split}.tar.gz"
            _logger.info("fetching %s (%d clip(s) wanted)", archive, len(names))
            try:
                local = hub.hf_hub_download(
                    self._hf_dataset, archive, repo_type="dataset", revision=None
                )
            except Exception as exc:  # hub raises many library-specific types
                raise DataError(f"bigos: could not fetch {archive!r}: {exc}") from exc
            with tarfile.open(local, "r:gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    base = member.name.replace("\\", "/").rsplit("/", 1)[-1]
                    if base not in names or base in extracted:
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    (clips_dir / base).write_bytes(source.read())
                    extracted.add(base)
            _logger.info("%s: %d/%d clip(s) extracted", subset, len(extracted), len(ordered))

        missing = [name for _, _, name in ordered if name not in extracted]
        if missing:
            _logger.warning(
                "%d clip(s) named in the parquet were absent from the tarballs (e.g. %s)",
                len(missing),
                missing[0],
            )

        manifest_path = root / "manifest.csv"
        audio_header = ["id", "text", "speaker_id", "subset", "audio_path"]
        retained = self._rows_to_retain(manifest_path, audio_header)
        rows = [item for item in ordered if item[2] in extracted]

        def build_row(
            item: tuple[Sample, str, str], _index: int
        ) -> tuple[list[Any], list[Path]] | None:
            sample, subset, clip_name = item
            row: list[Any] = [
                sample.id,
                sample.text,
                sample.speaker_id,
                subset,
                f"{CLIPS_DIR}/{clip_name}",
            ]
            return row, [clips_dir / clip_name]

        count = stream_records_to_manifest(
            manifest_path,
            iter(rows),
            build_row,
            header=audio_header,
            limit=None,
            source="bigos",
            empty_error="bigos audio download extracted no clips",
            progress_every=5000,
        )
        self._append_retained(manifest_path, retained)
        _logger.info("bigos audio download complete: %d clip(s) -> %s", count, clips_dir)

    def _load_from_hub(self, limit: int | None = None) -> Iterator[Sample]:
        """Stream text-only samples from the Hub."""
        count = 0
        for index, record in enumerate(self._stream_records()):
            if limit is not None and count >= limit:
                break
            sample = self._record_to_sample(record, index)
            if sample is None:
                continue
            count += 1
            yield sample

    def _stream_records(self) -> Iterator[dict[str, Any]]:
        """Stream raw Hub records (lazy ``datasets`` import).

        Records come from the parquet conversion rather than the dataset branch
        (see :data:`PARQUET_REVISION`). That conversion carries no ``audio``
        column, so clips are fetched separately from the release tarballs; see
        :meth:`_download_with_audio`.

        When ``subsets`` is configured, records from other source corpora are
        skipped here, which is what keeps an audio fetch to a few tractable
        tarballs instead of the full 28 GB.
        """
        datasets = optional_import(
            "datasets", extra="hf", purpose="loading BIGOS from the Hugging Face Hub"
        )
        _logger.info(
            "streaming %s (config=%s, split=%s) from the Hugging Face Hub",
            self._hf_dataset,
            self._hf_config,
            self._split,
        )
        try:
            stream = datasets.load_dataset(
                self._hf_dataset,
                self._hf_config,
                split=self._split,
                streaming=True,
                revision=self._revision,
            )
        except Exception as exc:  # datasets raises many library-specific types
            message = f"bigos: could not load {self._hf_dataset!r} from the Hub: {exc}"
            lowered = str(exc).lower()
            if "script" in lowered and "no longer supported" in lowered:
                message += (
                    "; BIGOS is published as a loading script and datasets>=4 removed "
                    "script support. Stream the Hub's parquet conversion instead by "
                    f"passing revision={PARQUET_REVISION!r} (this loader's default)"
                )
            elif "gated" in lowered or "authenticat" in lowered or "401" in lowered:
                message += (
                    "; this dataset is gated on the Hugging Face Hub: "
                    f"(1) sign in at https://huggingface.co/datasets/{self._hf_dataset} "
                    "and accept the access conditions, (2) authenticate locally with "
                    "`hf auth login` or by setting the HF_TOKEN environment variable, "
                    "then (3) re-run `tulip data download bigos`"
                )
            raise DataError(message) from exc
        for record in stream:
            if self._subsets and str(record.get("dataset") or "") not in self._subsets:
                continue
            yield record

    def _record_to_sample(self, record: dict[str, Any], index: int) -> Sample | None:
        """Convert one Hub record to a text-only Sample."""
        text = next(
            (str(record[f]).strip() for f in _TEXT_FIELDS if record.get(f)),
            "",
        )
        if not text:
            return None
        speaker = next(
            (str(record[f]).strip() for f in _SPEAKER_FIELDS if record.get(f)),
            "",
        )
        subset = str(record.get("dataset") or self._hf_config or "bigos")
        if not speaker:
            # Group by source subset: over-grouping is the safe direction for
            # speaker-disjoint splitting (see tulip.data.manifest).
            speaker = surrogate_speaker_id("bigos", subset)
        audio_name = str(record.get("audioname") or record.get("id") or index)
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"audio", *_TEXT_FIELDS} and isinstance(value, (str, int, float))
        }
        return Sample(
            id=f"bigos-{subset}-{audio_name}",
            text=text,
            speaker_id=speaker,
            source="bigos",
            metadata=metadata,
        )


__all__ = ["BigosLoader"]
