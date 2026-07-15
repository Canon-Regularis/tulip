"""Multi-track leaderboards: one ranking per input modality.

The benchmark covers three input modalities: written text, raw audio, and the
text transcribed from speech. Ranking them all in one table is misleading,
because a text model and an audio model never compete on the same input. A track
is one such competition: a named set of experiments and competitor models that do
share an input modality, ranked among themselves.

A track is declarative, not behavioural, so it is a pydantic model rather than a
registry: a :class:`TrackedSuite` lists its :class:`Track` entries, each of which
is a :class:`~tulip.evaluation.leaderboard.LeaderboardSuite` in all but name plus
the modality it covers. Running and rendering reuse the leaderboard machinery
unchanged, so a track's ranking is deterministic and byte-stable exactly as the
single-table leaderboard is. The combined artifact keys every row by ``(track,
experiment, model)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import TaskType
from tulip.evaluation.leaderboard import (
    LeaderboardSuite,
    render_leaderboard_markdown,
    run_leaderboard,
)
from tulip.utils.io import read_yaml
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.evaluation.benchmark import BenchmarkResult

_logger = get_logger(__name__)

#: Artifact name for the combined multi-track ranking (deterministic).
TRACKS_MD = "leaderboard-tracks.md"

__all__ = [
    "TRACKS_MD",
    "Track",
    "TrackResult",
    "TrackedSuite",
    "load_tracked_suite",
    "render_tracked_markdown",
    "run_tracked_leaderboard",
    "write_tracked_leaderboard",
]


class Track(BaseModel):
    """One modality-homogeneous competition within a tracked leaderboard.

    Attributes:
        name: Track identifier (e.g. ``text``, ``audio``, ``transcribed_speech``),
            used as the section heading and the per-track subdirectory.
        task: The input modality every experiment in the track shares. Text and
            transcribed speech both use :attr:`~tulip.core.types.TaskType.TEXT`;
            the track name carries the distinction the task type cannot.
        configs: Experiment YAMLs benchmarked in this track.
        models: Competitor models applied to every config (empty uses each
            config's own model).
        calibration_bins: Calibration bin count, or ``None`` to leave it off.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    task: TaskType
    configs: list[Path]
    models: list[str] = Field(default_factory=list)
    calibration_bins: int | None = Field(default=None, ge=1)

    def as_suite(self) -> LeaderboardSuite:
        """View this track as a :class:`LeaderboardSuite` for the existing runner."""
        return LeaderboardSuite(
            name=self.name,
            configs=self.configs,
            models=self.models,
            calibration_bins=self.calibration_bins,
        )


class TrackedSuite(BaseModel):
    """A leaderboard split into per-modality tracks.

    Attributes:
        name: Suite identifier, used as the artifact root and echoed into output.
        tracks: The tracks, ranked and rendered in declared order.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    tracks: list[Track] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_duplicate_track_names(self) -> TrackedSuite:
        # A track name is both a section heading and a per-track subdirectory, so
        # a duplicate would silently overwrite one track's dump with another's.
        names = [track.name for track in self.tracks]
        if len(names) != len(set(names)):
            duplicated = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"track names must be unique; duplicated: {duplicated}")
        return self


@dataclass(frozen=True)
class TrackResult:
    """One track's benchmark results, paired with the track that produced them."""

    track: Track
    results: tuple[BenchmarkResult, ...]


def load_tracked_suite(path: Path | str) -> TrackedSuite:
    """Load and validate a :class:`TrackedSuite` from a YAML file.

    Raises:
        ConfigurationError: if the file is missing, unparsable, not a mapping, or
            fails schema validation.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"tracked suite file not found: {path}")
    try:
        raw = read_yaml(path)
    except Exception as exc:
        raise ConfigurationError(f"could not parse YAML suite {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"suite {path} must be a YAML mapping, got {type(raw).__name__}")
    try:
        return TrackedSuite.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid tracked suite {path}:\n{exc}") from exc


def run_tracked_leaderboard(suite: TrackedSuite) -> list[TrackResult]:
    """Run every track's ``(config, model)`` pairs on their frozen splits.

    Each track is run through the untouched
    :func:`~tulip.evaluation.leaderboard.run_leaderboard`, so competitors within a
    track see byte-identical speaker-disjoint splits.

    Returns:
        One :class:`TrackResult` per track, in declared order.
    """
    track_results: list[TrackResult] = []
    for track in suite.tracks:
        _logger.info("tracked leaderboard %r: running track %r", suite.name, track.name)
        results = run_leaderboard(track.as_suite())
        track_results.append(TrackResult(track=track, results=tuple(results)))
    return track_results


def render_tracked_markdown(track_results: Sequence[TrackResult], *, name: str) -> str:
    """Render the combined multi-track ranking as deterministic markdown.

    Each track becomes its own ranked section (reusing
    :func:`~tulip.evaluation.leaderboard.render_leaderboard_markdown`), so the
    combined document is byte-stable when its inputs are.
    """
    parts = [f"# Leaderboard: {name}"]
    for track_result in track_results:
        track = track_result.track
        parts.append(f"## Track: {track.name} ({track.task.value})")
        parts.append(render_leaderboard_markdown(track_result.results))
    return "\n\n".join(parts)


def write_tracked_leaderboard(
    track_results: Sequence[TrackResult], out_dir: Path | str, *, suite: TrackedSuite
) -> None:
    """Write the combined ranking plus a per-track raw dump.

    Emits ``leaderboard-tracks.md`` (deterministic, the combined ranking) at the
    root, and one ``<track>/leaderboard.json`` per track (the raw benchmark dump,
    which retains machine-dependent timings and is therefore not byte-guaranteed).
    """
    from tulip.evaluation.benchmark import save_benchmark
    from tulip.evaluation.leaderboard import LEADERBOARD_JSON

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_tracked_markdown(track_results, name=suite.name)
    (out_dir / TRACKS_MD).write_text(markdown + "\n", encoding="utf-8", newline="\n")

    for track_result in track_results:
        track_dir = out_dir / track_result.track.name
        track_dir.mkdir(parents=True, exist_ok=True)
        save_benchmark(list(track_result.results), track_dir / LEADERBOARD_JSON)
    _logger.info("wrote tracked leaderboard artifacts to %s", out_dir)
