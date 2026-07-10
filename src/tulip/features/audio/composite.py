"""Compose configured audio feature extractors into one sklearn FeatureUnion.

Mirrors ``tulip.features.text.build_text_features``: experiment configs list
audio features as :class:`~tulip.config.schemas.ComponentConfig` entries
(registry name + params), and :func:`build_audio_features` resolves them
through :data:`tulip.features.AUDIO_FEATURES` into a single transformer whose
output rows concatenate every configured feature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.features._composite import build_feature_union
from tulip.features.registries import AUDIO_FEATURES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sklearn.pipeline import FeatureUnion

    from tulip.config.schemas import ComponentConfig

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
    return build_feature_union(AUDIO_FEATURES, configs)
