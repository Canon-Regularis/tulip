"""Shared YAML-to-pydantic loader for the benchmark suites.

The leaderboard suite and the tracked suite are each a YAML file validated into a
pydantic model, and their loaders were character-for-character identical apart
from the model class and the noun in each error message. This holds that scaffold
once. The frozen :mod:`tulip.config.loader` keeps its own copy for
``load_experiment_config``: it cannot import this without editing a frozen module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from tulip.core.exceptions import ConfigurationError
from tulip.utils.io import read_yaml

__all__ = ["load_yaml_model"]

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml_model(path: Path | str, model: type[ModelT], *, noun: str) -> ModelT:
    """Load ``path`` as YAML and validate it into the pydantic ``model``.

    The shared scaffold behind the suite loaders: confirm the file exists, parse
    the YAML, require a top-level mapping, and validate it into ``model``, with
    ``noun`` naming the artifact in every error message.

    Args:
        path: The YAML file to load.
        model: The pydantic model class to validate into.
        noun: A short name for the artifact, e.g. ``"leaderboard suite"``; it
            names the file in every error message.

    Returns:
        The validated ``model`` instance.

    Raises:
        ConfigurationError: if the file is missing, unparsable, not a mapping, or
            fails schema validation.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"{noun} file not found: {path}")
    try:
        raw = read_yaml(path)
    except Exception as exc:
        raise ConfigurationError(f"could not parse YAML {noun} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{noun} {path} must be a YAML mapping, got {type(raw).__name__}")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid {noun} {path}:\n{exc}") from exc
