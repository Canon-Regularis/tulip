"""Compose configured audio feature extractors into one sklearn FeatureUnion.

Mirrors ``tulip.features.text.build_text_features``: experiment configs list
audio features as :class:`~tulip.config.schemas.ComponentConfig` entries
(registry name + params), and :func:`build_audio_features` resolves them
through :data:`tulip.features.AUDIO_FEATURES` into a single transformer whose
output rows concatenate every configured feature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.pipeline import FeatureUnion

from tulip.config.schemas import ComponentConfig
from tulip.core.exceptions import ConfigurationError
from tulip.features.registries import AUDIO_FEATURES
from tulip.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["build_audio_features"]


def build_audio_features(
    configs: Sequence[ComponentConfig | Mapping[str, Any] | str],
) -> FeatureUnion:
    """Build a :class:`~sklearn.pipeline.FeatureUnion` of audio features.

    Args:
        configs: Feature specifications. Each entry may be a
            :class:`ComponentConfig`, a mapping with ``name``/``params`` keys,
            or a bare registry name (default params). Repeated names get
            numeric suffixes so FeatureUnion step names stay unique.

    Returns:
        A FeatureUnion whose transform concatenates all configured features,
        one row per audio file path.

    Raises:
        ConfigurationError: If ``configs`` is empty or an entry is malformed.
        UnknownComponentError: If a name is not in ``AUDIO_FEATURES``.
    """
    if not configs:
        raise ConfigurationError("build_audio_features requires at least one feature config")
    steps: list[tuple[str, Any]] = []
    used_names: set[str] = set()
    for entry in configs:
        config = _coerce_config(entry)
        extractor = AUDIO_FEATURES.create(config.name, **config.params)
        step_name = config.name
        suffix = 2
        while step_name in used_names:
            step_name = f"{config.name}_{suffix}"
            suffix += 1
        used_names.add(step_name)
        steps.append((step_name, extractor))
    logger.debug("built audio FeatureUnion with steps: %s", [name for name, _ in steps])
    return FeatureUnion(steps)


def _coerce_config(entry: ComponentConfig | Mapping[str, Any] | str) -> ComponentConfig:
    """Normalise a config entry into a :class:`ComponentConfig`."""
    if isinstance(entry, ComponentConfig):
        return entry
    if isinstance(entry, str):
        return ComponentConfig(name=entry)
    if isinstance(entry, Mapping):
        try:
            return ComponentConfig.model_validate(dict(entry))
        except ValueError as exc:
            raise ConfigurationError(f"invalid audio feature config {entry!r}: {exc}") from exc
    raise ConfigurationError(
        f"cannot interpret audio feature config of type {type(entry).__name__}: {entry!r}"
    )
