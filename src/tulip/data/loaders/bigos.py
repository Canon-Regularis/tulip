"""Loader for BIGOS (https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from tulip.core.exceptions import DataError
from tulip.core.types import Sample
from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.manifest import surrogate_speaker_id
from tulip.data.registry import DATASETS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

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
      *text-only* samples -- decoding hub-hosted audio to local files is out
      of scope for a loader; audio experiments should download the clips and
      use the manifest mode.

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
        yield from self._load_from_hub()

    def _load_from_hub(self) -> Iterator[Sample]:
        """Stream text-only samples from the Hub (lazy ``datasets`` import)."""
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
            raise DataError(
                f"bigos: could not load {self._hf_dataset!r} from the Hub: {exc}"
            ) from exc

        count = 0
        for index, record in enumerate(stream):
            if self._limit is not None and count >= self._limit:
                break
            sample = self._record_to_sample(record, index)
            if sample is None:
                continue
            count += 1
            yield sample

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
