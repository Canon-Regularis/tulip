"""Shared machinery for composing registered extractors into a FeatureUnion.

Both :func:`tulip.features.text.composite.build_text_features` and
:func:`tulip.features.audio.composite.build_audio_features` delegate here, so
config coercion, step naming, and error behaviour are identical for the two
modalities. Previously each had its own copy with silently divergent policies
(different step-name normalisation, only one accepting bare-string entries).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sklearn.pipeline import FeatureUnion

from tulip.config.schemas import ComponentConfig
from tulip.core.exceptions import ConfigurationError
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from tulip.core.registry import Registry

logger = get_logger(__name__)

__all__ = ["build_feature_union"]

ConfigEntry = ComponentConfig | Mapping[str, Any] | str


def build_feature_union(registry: Registry[Any], configs: Sequence[ConfigEntry]) -> FeatureUnion:
    """Build a :class:`~sklearn.pipeline.FeatureUnion` from component configs.

    Args:
        registry: The feature registry to resolve names against.
        configs: Feature specifications; each entry may be a
            :class:`ComponentConfig`, a mapping with ``name``/``params`` keys
            (as parsed from YAML), or a bare registry name (default params).

    Returns:
        An unfitted FeatureUnion over the configured extractors, in order.
        Step names are normalised (lowercase, dashes to underscores, matching
        registry normalisation); repeated components get numeric suffixes so
        FeatureUnion names stay unique.

    Raises:
        ConfigurationError: If ``configs`` is empty or an entry is malformed.
        UnknownComponentError: If a name is not registered (the error lists
            close-match suggestions).
    """
    if not configs:
        raise ConfigurationError(
            f"at least one {registry.kind} config is required to build a feature union"
        )
    steps: list[tuple[str, Any]] = []
    used_names: set[str] = set()
    for entry in configs:
        config = _coerce_config(registry, entry)
        try:
            extractor = registry.create(config.name, **config.params)
        except (TypeError, ValueError) as exc:
            # A mistyped params entry reaches the extractor constructor as an
            # unexpected keyword; name the feature, don't leak a raw traceback.
            raise ConfigurationError(
                f"cannot build {registry.kind} {config.name!r} from its configured "
                f"params {dict(config.params)!r}: {exc}"
            ) from exc
        base = config.name.strip().lower().replace("-", "_")
        step_name = base
        suffix = 2
        while step_name in used_names:
            step_name = f"{base}_{suffix}"
            suffix += 1
        used_names.add(step_name)
        steps.append((step_name, extractor))
    logger.debug("built %s FeatureUnion with steps: %s", registry.kind, [name for name, _ in steps])
    return FeatureUnion(steps)


def _coerce_config(registry: Registry[Any], entry: ConfigEntry) -> ComponentConfig:
    """Normalise a config entry into a :class:`ComponentConfig`."""
    if isinstance(entry, ComponentConfig):
        return entry
    if isinstance(entry, str):
        return ComponentConfig(name=entry)
    if isinstance(entry, Mapping):
        try:
            return ComponentConfig.model_validate(dict(entry))
        except ValueError as exc:
            raise ConfigurationError(f"invalid {registry.kind} config {entry!r}: {exc}") from exc
    raise ConfigurationError(
        f"cannot interpret {registry.kind} config of type {type(entry).__name__}: {entry!r}"
    )
