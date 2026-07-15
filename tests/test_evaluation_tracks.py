"""Tests for tulip.evaluation.tracks (multi-track leaderboards)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import TaskType
from tulip.evaluation.benchmark import BenchmarkResult
from tulip.evaluation.leaderboard import LEADERBOARD_JSON, LeaderboardSuite
from tulip.evaluation.metrics import compute_metrics
from tulip.evaluation.tracks import (
    TRACKS_MD,
    Track,
    TrackedSuite,
    TrackResult,
    load_tracked_suite,
    render_tracked_markdown,
    write_tracked_leaderboard,
)

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG_YAML = """\
name: {name}
seed: 7
task: text
target: dialect
data:
  root: data/raw
  datasets:
    - name: synthetic
features:
  - name: char_tfidf
model:
  name: logistic_regression
output_dir: {output_dir}
"""


def _tracked_suite_yaml(name: str, config_path: Path, model: str) -> str:
    return (
        f"name: {name}\n"
        "tracks:\n"
        "  - name: text\n"
        "    task: text\n"
        f"    configs: [{config_path.as_posix()}]\n"
        f"    models: [{model}]\n"
    )


def _result(experiment: str, model: str, errors: int) -> BenchmarkResult:
    y_true = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y_pred = list(y_true)
    for index in range(errors):
        y_pred[index] = "b" if y_true[index] == "a" else "a"
    report = compute_metrics(y_true, y_pred)
    return BenchmarkResult(experiment=experiment, model=model, reports={"test": report}, n_train=50)


def _track_results() -> list[TrackResult]:
    text = Track(name="text", task=TaskType.TEXT, configs=[], models=["logistic_regression"])
    audio = Track(name="audio", task=TaskType.AUDIO, configs=[], models=["wav2vec2"])
    return [
        TrackResult(track=text, results=(_result("e-text", "logistic_regression", 0),)),
        TrackResult(track=audio, results=(_result("e-audio", "wav2vec2", 2),)),
    ]


# --------------------------------------------------------------- validation


def test_tracked_suite_requires_at_least_one_track() -> None:
    with pytest.raises(ValidationError):
        TrackedSuite(name="empty", tracks=[])


def test_tracked_suite_rejects_duplicate_track_names() -> None:
    # A duplicate name would overwrite one track's per-track dump with another's.
    text_a = Track(name="text", task=TaskType.TEXT, configs=[])
    text_b = Track(name="text", task=TaskType.TEXT, configs=[])
    with pytest.raises(ValidationError, match="unique"):
        TrackedSuite(name="dupes", tracks=[text_a, text_b])


def test_track_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Track.model_validate({"name": "text", "task": "text", "configs": [], "bogus": 1})


def test_track_as_suite_preserves_fields() -> None:
    track = Track(
        name="text",
        task=TaskType.TEXT,
        configs=[],
        models=["naive_bayes"],
        calibration_bins=10,
    )
    suite = track.as_suite()
    assert isinstance(suite, LeaderboardSuite)
    assert suite.name == "text"
    assert suite.models == ["naive_bayes"]
    assert suite.calibration_bins == 10


# --------------------------------------------------------------- render / write


def test_render_has_a_section_per_track_and_is_deterministic() -> None:
    track_results = _track_results()
    markdown = render_tracked_markdown(track_results, name="demo")
    assert "# Leaderboard: demo" in markdown
    assert "## Track: text (text)" in markdown
    assert "## Track: audio (audio)" in markdown
    assert markdown.index("Track: text") < markdown.index("Track: audio")  # declared order
    assert render_tracked_markdown(track_results, name="demo") == markdown


def test_write_emits_combined_markdown_and_per_track_json(tmp_path: Path) -> None:
    suite = TrackedSuite(
        name="demo",
        tracks=[tr.track for tr in _track_results()],
    )
    write_tracked_leaderboard(_track_results(), tmp_path, suite=suite)
    assert (tmp_path / TRACKS_MD).is_file()
    assert (tmp_path / "text" / LEADERBOARD_JSON).is_file()
    assert (tmp_path / "audio" / LEADERBOARD_JSON).is_file()


# --------------------------------------------------------------- loading


def test_load_tracked_suite_reads_a_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "tracks.yaml"
    path.write_text(
        "name: t\n"
        "tracks:\n"
        "  - name: text\n"
        "    task: text\n"
        "    configs: []\n"
        "    models: [naive_bayes]\n",
        encoding="utf-8",
    )
    suite = load_tracked_suite(path)
    assert suite.name == "t"
    assert suite.tracks[0].task is TaskType.TEXT


def test_load_tracked_suite_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_tracked_suite(tmp_path / "nope.yaml")


def test_load_tracked_suite_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_tracked_suite(path)


# --------------------------------------------------------------- integration


def test_run_tracked_leaderboard_over_a_real_config(tmp_path: Path) -> None:
    from tulip.evaluation.tracks import run_tracked_leaderboard

    config_path = tmp_path / "text.yaml"
    config_path.write_text(
        _CONFIG_YAML.format(name="synth-text", output_dir=(tmp_path / "artifacts").as_posix()),
        encoding="utf-8",
    )
    suite = TrackedSuite(
        name="one-track",
        tracks=[
            Track(name="text", task=TaskType.TEXT, configs=[config_path], models=["naive_bayes"])
        ],
    )
    track_results = run_tracked_leaderboard(suite)
    assert len(track_results) == 1
    assert track_results[0].track.name == "text"
    assert len(track_results[0].results) == 1
    assert track_results[0].results[0].model == "naive_bayes"


def test_cli_leaderboard_tracks(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tulip.cli.app import app

    config_path = tmp_path / "text.yaml"
    config_path.write_text(
        _CONFIG_YAML.format(name="synth-text", output_dir=(tmp_path / "artifacts").as_posix()),
        encoding="utf-8",
    )
    suite_path = tmp_path / "tracks.yaml"
    suite_path.write_text(
        _tracked_suite_yaml("cli-tracks", config_path, "naive_bayes"), encoding="utf-8"
    )
    result = CliRunner().invoke(
        app, ["leaderboard", str(suite_path), "--tracks", "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "cli-tracks" / TRACKS_MD).is_file()
