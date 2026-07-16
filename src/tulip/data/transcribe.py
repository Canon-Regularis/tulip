"""Transcribe audio samples into text: the bridge to the transcribed-speech track.

The benchmark names three input modalities: written text, raw audio, and text
transcribed from speech. This module produces the third. It decodes each
sample's audio through the canonical shared loader, runs a Whisper model over
it, and returns a copy of the sample carrying the transcript as ``text``. The
copies keep their ``audio_path`` and their labels, so one corpus can then feed
both the audio track and the transcribed-speech track.

Transcription is expensive and model-dependent, so results go through a
content-addressed cache keyed by the audio bytes, the checkpoint, and the
language, mirroring the LLM baseline's response cache. A cached corpus
re-transcribes for free and offline, and the cache key changes whenever the
audio or the model does.

The Whisper engine is injectable. Tests pass a fake callable; the default is
built lazily from the ``speech`` extra, so importing this module never
requires torch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from tulip._serialize import write_sorted_json
from tulip.utils.io import write_jsonl
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from tulip.core.types import Sample

__all__ = [
    "DEFAULT_CHECKPOINT",
    "TranscribeConfig",
    "TranscriptCache",
    "transcribe_samples",
    "write_transcribed_manifest",
]

_logger = get_logger(__name__)

#: Default ASR checkpoint; the same Whisper family the speech classifiers use.
DEFAULT_CHECKPOINT = "openai/whisper-small"

#: Audio is decoded at this rate for the ASR model (Whisper expects 16 kHz).
_ASR_SAMPLE_RATE = 16_000


class TranscribeConfig(BaseModel):
    """Parameters for one transcription run.

    A module-owned schema, not an extension of the frozen ``ExperimentConfig``.

    Attributes:
        checkpoint: Hugging Face ASR checkpoint to load.
        language: Language token forced during decoding, so short or noisy
            clips are not misdetected as another language.
        cache_dir: Directory backing the transcript cache; ``None`` caches only
            in memory for the lifetime of the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint: str = DEFAULT_CHECKPOINT
    language: str = "pl"
    cache_dir: Path | None = None


class TranscriptCache:
    """A content-addressed store of transcripts, keyed by audio and model.

    With a directory, entries persist as one small JSON file per key, so a
    second run over the same corpus is offline and free. Without one, entries
    live in memory only.
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, str] = {}

    @staticmethod
    def key(audio_bytes: bytes, *, checkpoint: str, language: str) -> str:
        """Content-address one clip under one model and language."""
        digest = hashlib.sha256()
        digest.update(audio_bytes)
        digest.update(checkpoint.encode("utf-8"))
        digest.update(language.encode("utf-8"))
        return digest.hexdigest()

    def get(self, key: str) -> str | None:
        """Return the cached transcript for ``key``, or ``None``."""
        if key in self._memory:
            return self._memory[key]
        if self.directory is None:
            return None
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        import json

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _logger.debug("unreadable transcript cache entry %s; ignoring", path)
            return None
        text = payload.get("text") if isinstance(payload, dict) else None
        if isinstance(text, str):
            self._memory[key] = text
            return text
        return None

    def put(self, key: str, text: str) -> None:
        """Store a transcript under ``key``."""
        self._memory[key] = text
        if self.directory is not None:
            write_sorted_json(self.directory / f"{key}.json", {"text": text})


def transcribe_samples(
    samples: Iterable[Sample],
    config: TranscribeConfig | None = None,
    *,
    asr: Callable[[Path], str] | None = None,
) -> list[Sample]:
    """Return text-carrying copies of every sample that has audio.

    Each copy keeps its ``audio_path`` and labels, gains the transcript as
    ``text``, and records the checkpoint and language in ``metadata`` so a
    transcribed corpus stays auditable. Samples without audio are skipped.

    Args:
        samples: The corpus to transcribe.
        config: Transcription parameters; defaults to :class:`TranscribeConfig`.
        asr: An ``audio path -> text`` callable to use instead of the default
            Whisper engine, which decodes through the canonical shared loader.

    Returns:
        The transcribed copies, in input order.

    Raises:
        MissingDependencyError: if the default engine is needed and the
            ``speech`` extra is not installed.
        DataError: if an audio file cannot be read.
    """
    config = config or TranscribeConfig()
    cache = TranscriptCache(config.cache_dir)
    engine = asr

    transcribed: list[Sample] = []
    skipped = 0
    for sample in samples:
        if sample.audio_path is None:
            skipped += 1
            continue
        audio_bytes = _read_audio_bytes(sample.audio_path)
        key = TranscriptCache.key(
            audio_bytes, checkpoint=config.checkpoint, language=config.language
        )
        text = cache.get(key)
        if text is None:
            if engine is None:
                engine = _build_whisper_asr(config)
            text = engine(Path(sample.audio_path))
            cache.put(key, text)
        transcribed.append(
            sample.model_copy(
                update={
                    "text": text,
                    "metadata": {
                        **sample.metadata,
                        "transcribed_by": config.checkpoint,
                        "transcription_language": config.language,
                    },
                }
            )
        )
    if skipped:
        _logger.info("transcribe: %d sample(s) without audio were skipped", skipped)
    return transcribed


def write_transcribed_manifest(samples: Sequence[Sample], root: Path | str) -> Path:
    """Persist transcribed samples as ``root/manifest.jsonl``.

    The manifest is written in the flat, one-object-per-line shape that
    :func:`tulip.data.manifest.read_manifest` consumes, so the transcribed
    corpus loads through the generic ``manifest`` loader and the text pipeline
    can train on it directly.

    Returns:
        The path to the written ``manifest.jsonl``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.jsonl"
    written = write_jsonl(path, (_to_manifest_record(sample) for sample in samples))
    _logger.info("wrote %d transcribed samples to %s", written, path)
    return path


def _to_manifest_record(sample: Sample) -> dict[str, object]:
    """Flatten one transcribed :class:`Sample` into a read_manifest record."""
    record: dict[str, object] = {
        "id": sample.id,
        "text": sample.text,
        "speaker_id": sample.speaker_id,
    }
    if sample.audio_path is not None:
        record["audio_path"] = Path(sample.audio_path).as_posix()
    for field in ("family", "dialect", "region", "village", "voivodeship"):
        value = getattr(sample.labels, field)
        if value is not None:
            record[field] = value
    for key in ("transcribed_by", "transcription_language"):
        if key in sample.metadata:
            record[key] = sample.metadata[key]
    return record


def _read_audio_bytes(path: Path | str) -> bytes:
    """The raw clip bytes, for content addressing."""
    from tulip.core.exceptions import DataError

    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise DataError(f"cannot read audio file {path}: {exc}") from exc


def _build_whisper_asr(config: TranscribeConfig) -> Callable[[Path], str]:
    """Build the default Whisper engine lazily (``speech`` extra).

    Audio decodes through the canonical shared loader, generation is greedy,
    and the language is forced, so a run is deterministic for a checkpoint.
    """
    transformers = optional_import("transformers", extra="speech", purpose="Whisper transcription")
    from tulip.features.audio.loading import load_audio

    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=config.checkpoint,
        generate_kwargs={
            "language": config.language,
            "task": "transcribe",
            "num_beams": 1,
            "do_sample": False,
        },
    )

    def asr(path: Path) -> str:
        signal = load_audio(path, sample_rate=_ASR_SAMPLE_RATE)
        result = pipe({"array": signal, "sampling_rate": _ASR_SAMPLE_RATE})
        return str(result["text"]).strip()

    return asr
