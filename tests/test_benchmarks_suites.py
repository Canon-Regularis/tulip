"""Every shipped leaderboard suite resolves: configs, datasets, features, models.

These suites are wired by hand in YAML, so a typo (a renamed feature, a dropped
dataset) would only surface at run time, and the real-text suite cannot run
without a manually-acquired corpus. This test validates the wiring statically:
each suite loads, every referenced config loads, and every dataset, feature,
and model name it names is registered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tulip.config import load_experiment_config
from tulip.data.registry import DATASETS
from tulip.evaluation.leaderboard import load_suite
from tulip.features.registries import AUDIO_FEATURES, TEXT_FEATURES
from tulip.models.registry import MODELS

_BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"

#: Every top-level suite YAML shipped under ``benchmarks/``.
_SUITES = sorted(_BENCHMARKS.glob("*suite*.yaml"))


def test_suite_files_are_discovered() -> None:
    # Guard against the glob silently matching nothing (e.g. a moved directory).
    names = {path.name for path in _SUITES}
    assert {"suite.yaml", "real_text_suite.yaml"} <= names


@pytest.mark.parametrize("suite_path", _SUITES, ids=lambda path: path.stem)
def test_suite_wiring_resolves(suite_path: Path) -> None:
    suite = load_suite(suite_path)
    assert suite.configs, f"{suite_path.name} lists no configs"

    for model_name in suite.models:
        MODELS.get(model_name)  # raises UnknownComponentError if unregistered

    for config_path in suite.configs:
        config = load_experiment_config(config_path)
        for dataset in config.data.datasets:
            DATASETS.get(dataset.name)
        registry = AUDIO_FEATURES if config.task.value == "audio" else TEXT_FEATURES
        for feature in config.features:
            registry.get(feature.name)


def test_real_text_suite_targets_dialektarium_and_dgp() -> None:
    # The real-text track's whole point is the two real prose corpora; keep the
    # wiring honest so the docs' "acquire these two" instruction stays accurate.
    suite = load_suite(_BENCHMARKS / "real_text_suite.yaml")
    corpora = {
        dataset.name
        for config_path in suite.configs
        for dataset in load_experiment_config(config_path).data.datasets
    }
    assert corpora == {"dialektarium", "dgp"}
