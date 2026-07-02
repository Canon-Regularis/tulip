"""Compose registered text feature extractors into one sklearn transformer.

``build_text_features`` turns a list of :class:`ComponentConfig` entries
(registry name + params, exactly as they appear in experiment YAML) into a
single :class:`sklearn.pipeline.FeatureUnion`, so classical models can consume
any mix of TF-IDF, stylometric, affix, and keyword features as one matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.pipeline import FeatureUnion

from tulip.config.schemas import ComponentConfig
from tulip.core.exceptions import ConfigurationError
from tulip.features.registries import TEXT_FEATURES
from tulip.utils.logging import get_logger

__all__ = ["build_text_features"]

logger = get_logger(__name__)


def build_text_features(
    configs: Sequence[ComponentConfig | Mapping[str, Any]],
) -> FeatureUnion:
    """Build one transformer from text-feature component configs.

    Each config's ``name`` is resolved against the ``TEXT_FEATURES`` registry
    and instantiated with its ``params``. Repeated components (e.g. two
    ``char_tfidf`` blocks with different n-gram ranges) get numeric suffixes so
    FeatureUnion transformer names stay unique.

    Args:
        configs: Component references; plain mappings with ``name``/``params``
            keys (as parsed from YAML) are accepted and validated too.

    Returns:
        An unfitted :class:`FeatureUnion` over the configured extractors, in
        the given order.

    Raises:
        ConfigurationError: If ``configs`` is empty.
        UnknownComponentError: If a name is not registered in
            ``TEXT_FEATURES`` (the error lists close-match suggestions).
    """
    if not configs:
        raise ConfigurationError("build_text_features requires at least one feature config")
    transformers: list[tuple[str, Any]] = []
    used: dict[str, int] = {}
    for entry in configs:
        config = (
            entry if isinstance(entry, ComponentConfig) else ComponentConfig.model_validate(entry)
        )
        extractor = TEXT_FEATURES.create(config.name, **config.params)
        base = config.name.strip().lower().replace("-", "_")
        occurrence = used.get(base, 0) + 1
        used[base] = occurrence
        union_name = base if occurrence == 1 else f"{base}_{occurrence}"
        transformers.append((union_name, extractor))
    logger.debug(
        "built text FeatureUnion with %d extractors: %s",
        len(transformers),
        [name for name, _ in transformers],
    )
    return FeatureUnion(transformers)
