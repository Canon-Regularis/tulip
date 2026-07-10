"""Tests for tulip.evaluation.leaderboard (the deterministic public leaderboard).

Hermetic and fast: :class:`BenchmarkResult` objects are constructed directly
(no experiments are trained here) so the tests exercise rendering, persistence,
and the determinism guarantee in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tulip.evaluation.benchmark import BenchmarkResult, load_benchmark
from tulip.evaluation.leaderboard import (
    LEADERBOARD_JSON,
    LEADERBOARD_MD,
    PROVENANCE_JSON,
    LeaderboardSuite,
    load_suite,
    render_leaderboard_markdown,
    write_leaderboard,
)
from tulip.evaluation.metrics import compute_metrics

#: Repo root, resolved from this file so tests do not depend on the CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]

_EXPERIMENT_CONFIG_YAML = """\
name: test-track
seed: 7
task: text
target: dialect
data:
  root: data/raw
  datasets:
    - name: synthetic
model:
  name: logistic_regression
output_dir: {output_dir}
"""


def _result(model: str, errors: int, *, seconds: float, with_proba: bool = True) -> BenchmarkResult:
    """Build a result whose test-split quality decreases with ``errors`` (0..4).

    ``errors`` are distinct across the fixtures below, so macro-F1 is unique and
    the leaderboard sort is total (not merely stable).
    """
    y_true = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y_pred = list(y_true)
    for index in range(errors):
        y_pred[index] = "b" if y_true[index] == "a" else "a"
    proba = [[0.9, 0.1] if p == "a" else [0.1, 0.9] for p in y_pred] if with_proba else None
    reports = {
        split: compute_metrics(y_true, y_pred, y_proba=proba, metadata={"split": split})
        for split in ("validation", "test")
    }
    return BenchmarkResult(
        experiment="synthetic-char-baseline",
        model=model,
        target_level="dialect",
        reports=reports,
        wall_seconds=seconds,
        n_train=100,
        n_test=len(y_true),
    )


@pytest.fixture
def results() -> list[BenchmarkResult]:
    return [
        _result("mediocre", 2, seconds=5.0),
        _result("best", 0, seconds=60.0),
        _result("worst", 3, seconds=1.0, with_proba=False),
    ]


def _cells(row: str) -> list[str]:
    """Split a markdown table row into its trimmed cell values."""
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def test_render_contains_models_and_metric_headers(results: list[BenchmarkResult]) -> None:
    markdown = render_leaderboard_markdown(results)
    for model in ("best", "mediocre", "worst"):
        assert model in markdown
    headers = ("Experiment", "Model", "Accuracy", "F1 (macro)", "F1 (weighted)", "ROC AUC", "Train")
    for header in headers:
        assert header in markdown
    lines = markdown.splitlines()
    # Row identity is (experiment, model): a suite runs each model on every
    # config, so the model name alone does not identify a row.
    assert _cells(lines[2])[1] == "best"  # sorted best-first by macro-F1
    assert "| n/a |" in markdown  # the proba-less "worst" model has no ROC AUC


def test_render_identifies_rows_by_experiment_not_model_alone(
    results: list[BenchmarkResult],
) -> None:
    """Two experiments sharing a model name must not produce indistinguishable rows."""
    twin = results[0].model_copy(update={"experiment": "synthetic-lexical-baseline"})
    markdown = render_leaderboard_markdown([*results, twin])
    experiments = {_cells(line)[0] for line in markdown.splitlines()[2:] if line.startswith("|")}
    assert experiments == {"synthetic-char-baseline", "synthetic-lexical-baseline"}


def test_render_never_emits_wall_clock_time(results: list[BenchmarkResult]) -> None:
    """The whole point of the wrapper: no nondeterministic timing column."""
    markdown = render_leaderboard_markdown(results)
    lowered = markdown.lower()
    assert "seconds" not in lowered
    assert "wall" not in lowered


def test_render_is_deterministic_and_order_independent(results: list[BenchmarkResult]) -> None:
    first = render_leaderboard_markdown(results)
    assert render_leaderboard_markdown(results) == first  # byte-identical re-render
    assert render_leaderboard_markdown(list(reversed(results))) == first  # total ordering


def _suite_with_config(tmp_path: Path) -> LeaderboardSuite:
    """A suite pointing at a real (loadable) experiment config under ``tmp_path``.

    Its ``output_dir`` is inside ``tmp_path`` where no ``build_manifest.json``
    exists, so provenance exercises the manifest-absent path.
    """
    config_path = tmp_path / "track.yaml"
    config_path.write_text(
        _EXPERIMENT_CONFIG_YAML.format(output_dir=(tmp_path / "artifacts").as_posix()),
        encoding="utf-8",
    )
    return LeaderboardSuite(
        name="test-suite",
        configs=[config_path],
        models=["naive_bayes", "logistic_regression"],
    )


def test_write_leaderboard_emits_three_files(
    results: list[BenchmarkResult], tmp_path: Path
) -> None:
    suite = _suite_with_config(tmp_path)
    out_dir = tmp_path / "out"
    write_leaderboard(results, out_dir, suite=suite)

    assert (out_dir / LEADERBOARD_MD).is_file()
    assert (out_dir / LEADERBOARD_JSON).is_file()
    assert (out_dir / PROVENANCE_JSON).is_file()
    # leaderboard.json is the raw dump and round-trips via the benchmark loader.
    assert load_benchmark(out_dir / LEADERBOARD_JSON) == results


def test_provenance_has_sorted_keys_and_no_timestamp(
    results: list[BenchmarkResult], tmp_path: Path
) -> None:
    suite = _suite_with_config(tmp_path)
    out_dir = tmp_path / "out"
    write_leaderboard(results, out_dir, suite=suite)

    text = (out_dir / PROVENANCE_JSON).read_text(encoding="utf-8")
    provenance = json.loads(text)
    assert list(provenance) == sorted(provenance)  # top-level keys sorted

    lowered = text.lower()
    assert "timestamp" not in lowered
    assert "wall_seconds" not in lowered
    assert "seconds" not in lowered

    assert provenance["suite"] == "test-suite"
    assert provenance["models"] == ["logistic_regression", "naive_bayes"]  # sorted
    assert provenance["tulip_version"]
    (config_entry,) = provenance["configs"]
    assert config_entry["name"] == "test-track"
    assert config_entry["seed"] == 7
    assert config_entry["sizes"] is None  # no build manifest on disk
    assert config_entry["class_distribution"] is None
    # Every result is tagged with its experiment for disambiguation.
    assert {row["experiment"] for row in provenance["results"]} == {"synthetic-char-baseline"}


def test_provenance_is_byte_identical_across_runs(
    results: list[BenchmarkResult], tmp_path: Path
) -> None:
    suite = _suite_with_config(tmp_path)
    write_leaderboard(results, tmp_path / "a", suite=suite)
    write_leaderboard(list(reversed(results)), tmp_path / "b", suite=suite)

    first = (tmp_path / "a" / PROVENANCE_JSON).read_bytes()
    second = (tmp_path / "b" / PROVENANCE_JSON).read_bytes()
    assert first == second  # sorted output + total ordering => reorder-invariant
    md_first = (tmp_path / "a" / LEADERBOARD_MD).read_bytes()
    md_second = (tmp_path / "b" / LEADERBOARD_MD).read_bytes()
    assert md_first == md_second


def test_load_suite_round_trips_committed_file() -> None:
    suite = load_suite(REPO_ROOT / "benchmarks" / "suite.yaml")
    assert suite.name == "synthetic-leaderboard"
    assert "naive_bayes" in suite.models
    assert "logistic_regression" in suite.models
    # The referenced config files exist and parse as experiment configs.
    from tulip.config.loader import load_experiment_config

    assert suite.configs, "committed suite must reference at least one config"
    for config_path in suite.configs:
        config = load_experiment_config(REPO_ROOT / config_path)
        assert config.data.datasets[0].name == "synthetic"
