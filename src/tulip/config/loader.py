"""Loading and saving experiment configurations."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from tulip.config.schemas import ExperimentConfig
from tulip.core.exceptions import ConfigurationError
from tulip.utils.io import read_yaml, write_yaml


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Raises:
        ConfigurationError: if the file is missing, unparsable, or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"config file not found: {path}")
    try:
        raw = read_yaml(path)
    except Exception as exc:
        raise ConfigurationError(f"could not parse YAML config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"config {path} must be a YAML mapping, got {type(raw).__name__}")
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid experiment config {path}:\n{exc}") from exc


def save_experiment_config(config: ExperimentConfig, path: Path | str) -> None:
    """Serialise an experiment config to YAML (round-trips with the loader)."""
    write_yaml(Path(path), config.model_dump(mode="json"))
