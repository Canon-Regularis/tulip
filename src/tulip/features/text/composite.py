"""Compose registered text feature extractors into one sklearn transformer.

``build_text_features`` turns a list of :class:`ComponentConfig` entries
(registry name + params, exactly as they appear in experiment YAML) into a
single :class:`sklearn.pipeline.FeatureUnion`, so classical models can consume
any mix of TF-IDF, stylometric, affix, and keyword features as one matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.features._composite import build_feature_union
from tulip.features.registries import TEXT_FEATURES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sklearn.pipeline import FeatureUnion

    from tulip.config.schemas import ComponentConfig

__all__ = ["build_text_features"]


def build_text_features(
    configs: Sequence[ComponentConfig | Mapping[str, Any] | str],
) -> FeatureUnion:
    """Build one transformer from text-feature component configs.

    Each config's ``name`` is resolved against the ``TEXT_FEATURES`` registry
    and instantiated with its ``params``; bare registry names are accepted for
    default params. Repeated components (e.g. two ``char_tfidf`` blocks with
    different n-gram ranges) get numeric suffixes so FeatureUnion transformer
    names stay unique.

    Args:
        configs: Component references; plain mappings with ``name``/``params``
            keys (as parsed from YAML) and bare names are accepted too.

    Returns:
        An unfitted :class:`FeatureUnion` over the configured extractors, in
        the given order.

    Raises:
        ConfigurationError: If ``configs`` is empty or an entry is malformed.
        UnknownComponentError: If a name is not registered in
            ``TEXT_FEATURES`` (the error lists close-match suggestions).
    """
    return build_feature_union(TEXT_FEATURES, configs)
