"""Tests for tulip.config.loader error handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.config import load_experiment_config
from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_experiment_config(tmp_path / "nope.yaml")


def test_unparsable_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: [unclosed list\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="could not parse"):
        load_experiment_config(path)


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be a YAML mapping"):
        load_experiment_config(path)


def test_schema_invalid_config_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: incomplete\n", encoding="utf-8")  # missing required fields
    with pytest.raises(ConfigurationError, match="invalid experiment config"):
        load_experiment_config(path)
