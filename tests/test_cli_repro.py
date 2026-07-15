"""Tests for the reproducibility gate (tulip.cli._repro.verify_reproduction)."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from tulip.cli._repro import reproduce_from_scratch, verify_reproduction
from tulip.evaluation.leaderboard import (
    load_suite,
    run_leaderboard,
    write_leaderboard,
    write_significance,
)

if TYPE_CHECKING:
    from pathlib import Path

# These retrain the whole suite several times; keep them out of the fast gate.
# CI's dedicated `repro` job exercises the same determinism separately.
pytestmark = pytest.mark.slow

_CONFIG_YAML = """\
name: repro-test
seed: 42
task: text
target: dialect
data:
  root: data/raw
  datasets:
    - name: synthetic
      params:
        n_speakers_per_dialect: 6
        samples_per_speaker: 8
        include_standard: false
        seed: 7
  deduplicate: false
  min_text_chars: 10
features:
  - name: char_tfidf
model:
  name: logistic_regression
split:
  seed: 42
output_dir: {output_dir}
"""

_SUITE_YAML = """\
name: repro-test-suite
configs:
  - {config_path}
models:
  - naive_bayes
  - logistic_regression
calibration_bins: 10
"""


@pytest.fixture
def suite_and_committed(tmp_path: Path) -> tuple[Path, Path]:
    """Write a tiny synthetic suite and generate its committed artifacts.

    Returns ``(suite_path, committed_root)`` where the committed board lives at
    ``committed_root/<suite name>/``.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _CONFIG_YAML.format(output_dir=(tmp_path / "artifacts").as_posix()), encoding="utf-8"
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(_SUITE_YAML.format(config_path=config_path.as_posix()), encoding="utf-8")

    committed_root = tmp_path / "committed"
    suite = load_suite(suite_path)
    results = run_leaderboard(suite)
    destination = committed_root / suite.name
    write_leaderboard(results, destination, suite=suite)
    write_significance(results, destination)
    return suite_path, committed_root


def test_reproduces_the_committed_board(
    suite_and_committed: tuple[Path, Path], tmp_path: Path
) -> None:
    suite_path, committed_root = suite_and_committed
    drift = verify_reproduction(suite_path, committed_root, tmp_path / "work")
    assert drift == []


def test_from_scratch_reproduces_in_isolation(
    suite_and_committed: tuple[Path, Path], tmp_path: Path
) -> None:
    # Simulate a clean room (fresh checkout / container): remove the config's
    # declared output_dir entirely, so nothing pre-existing can be read. A
    # from-scratch run rebuilds everything under its own work dir and must still
    # reproduce the committed board, including the provenance sizes and dataset
    # digest, which it reads from the isolated build, not the deleted tree.
    suite_path, committed_root = suite_and_committed
    shutil.rmtree(tmp_path / "artifacts")  # the config's declared output_dir
    work = tmp_path / "scratch"
    drift = reproduce_from_scratch(suite_path, committed_root, work)
    assert drift == []
    assert (work / "build").is_dir()  # the build landed inside the isolated work dir
    # And provenance carried real sizes/digest, sourced from the isolated build.
    provenance = json.loads(
        (work / "repro-test-suite" / "provenance.json").read_text(encoding="utf-8")
    )
    (config_entry,) = provenance["configs"]
    assert config_entry["sizes"] is not None
    assert config_entry["dataset_digest"] is not None


def test_from_scratch_detects_drift(suite_and_committed: tuple[Path, Path], tmp_path: Path) -> None:
    suite_path, committed_root = suite_and_committed
    board = committed_root / "repro-test-suite" / "provenance.json"
    board.write_text('{"tampered": true}', encoding="utf-8")
    drift = reproduce_from_scratch(suite_path, committed_root, tmp_path / "scratch")
    assert any("provenance.json" in line for line in drift)


def test_detects_a_drifted_leaderboard(
    suite_and_committed: tuple[Path, Path], tmp_path: Path
) -> None:
    suite_path, committed_root = suite_and_committed
    board = committed_root / "repro-test-suite" / "leaderboard.md"
    board.write_text(board.read_text(encoding="utf-8") + "\ntampered\n", encoding="utf-8")

    drift = verify_reproduction(suite_path, committed_root, tmp_path / "work")
    assert any("leaderboard.md" in line for line in drift)


def test_detects_drifted_significance(
    suite_and_committed: tuple[Path, Path], tmp_path: Path
) -> None:
    suite_path, committed_root = suite_and_committed
    sig = committed_root / "repro-test-suite" / "significance-repro-test.json"
    sig.write_text('{"tampered": true}', encoding="utf-8")

    drift = verify_reproduction(suite_path, committed_root, tmp_path / "work")
    assert any("significance-repro-test.json" in line for line in drift)


def test_missing_committed_directory_is_drift(
    suite_and_committed: tuple[Path, Path], tmp_path: Path
) -> None:
    suite_path, _ = suite_and_committed
    drift = verify_reproduction(suite_path, tmp_path / "does-not-exist", tmp_path / "work")
    assert drift and "no committed artifacts" in drift[0]
