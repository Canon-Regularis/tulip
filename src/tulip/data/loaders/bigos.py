"""Loader for BIGOS (https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.core.types import Sample
from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.loaders._hub_audio import CLIPS_DIR, write_hub_clip
from tulip.data.loaders._stream import stream_records_to_manifest
from tulip.data.manifest import surrogate_speaker_id
from tulip.data.registry import DATASETS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_logger = get_logger(__name__)

#: Field names probed for a speaker identifier in Hub records.
_SPEAKER_FIELDS = ("speaker_id", "speakerid", "speaker", "client_id")

#: Field names probed for the transcription text in Hub records.
_TEXT_FIELDS = ("ref_orig", "ref", "transcription", "sentence", "text")


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
        hf_dataset: Hub dataset id for hub mode.
        hf_config: Optional Hub configuration (source-corpus subset) name.
        split: Hub split to stream (default ``"train"``).
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
        "conditions at https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos, "
        "authenticate (`hf auth login` or HF_TOKEN), then `tulip data download "
        "bigos` streams the transcriptions into data/raw/bigos/manifest.csv "
        "(text-only; requires the `hf` extra)"
    )

    def __init__(
        self,
        manifest: str | None = None,
        *,
        from_hub: bool = False,
        hf_dataset: str = "michaljunczyk/pl-asr-bigos",
        hf_config: str | None = None,
        split: str = "train",
        limit: int | None = None,
    ) -> None:
        super().__init__(manifest)
        self._from_hub = from_hub
        self._hf_dataset = hf_dataset
        self._hf_config = hf_config
        self._split = split
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
                ``audio`` also fetches the clips (their original encoded bytes,
                so no audio extra is needed).

        Raises:
            ConfigurationError: on unknown options.
            DataError: if the Hub yields no samples, or a record lacks audio
                in audio mode.
            MissingDependencyError: without the ``hf`` extra installed.
        """
        limit = options.pop("limit", self._limit)
        audio = bool(options.pop("audio", False))
        if options:
            raise ConfigurationError(
                f"bigos download got unknown option(s): {', '.join(sorted(options))}"
            )
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.csv"
        header = ["id", "text", "speaker_id", "subset"]
        if audio:
            header.append("audio_path")

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
            clips: list[Path] = []
            if audio:
                # The stream index makes the clip name unique even when two sample
                # ids sanitise to the same filesystem-safe stem.
                clip = write_hub_clip(
                    record.get("audio"), root / CLIPS_DIR, f"{index:08d}-{sample.id}"
                )
                clips.append(root / CLIPS_DIR / clip)
                row.append(f"{CLIPS_DIR}/{clip}")
            return row, clips

        count = stream_records_to_manifest(
            manifest_path,
            self._stream_records(raw_audio=audio),
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
        _logger.info("bigos download complete: %d samples -> %s", count, manifest_path)

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

    def _stream_records(self, *, raw_audio: bool = False) -> Iterator[dict[str, Any]]:
        """Stream raw Hub records (lazy ``datasets`` import).

        With ``raw_audio``, the ``audio`` column is cast to ``Audio(decode=False)``
        so each record carries the clip's original encoded bytes, which the audio
        fetch writes verbatim. Without it (text mode), the column is left as the
        Hub serves it and never touched.
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
                self._hf_dataset, self._hf_config, split=self._split, streaming=True
            )
        except Exception as exc:  # datasets raises many library-specific types
            message = f"bigos: could not load {self._hf_dataset!r} from the Hub: {exc}"
            lowered = str(exc).lower()
            if "gated" in lowered or "authenticat" in lowered or "401" in lowered:
                message += (
                    "; this dataset is gated on the Hugging Face Hub: "
                    f"(1) sign in at https://huggingface.co/datasets/{self._hf_dataset} "
                    "and accept the access conditions, (2) authenticate locally with "
                    "`hf auth login` or by setting the HF_TOKEN environment variable, "
                    "then (3) re-run `tulip data download bigos`"
                )
            raise DataError(message) from exc
        if raw_audio:
            stream = stream.cast_column("audio", datasets.Audio(decode=False))
        yield from stream

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
