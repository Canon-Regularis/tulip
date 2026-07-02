"""Feature extraction for text and audio.

Importing this package registers all built-in feature extractors in
:data:`TEXT_FEATURES` and :data:`AUDIO_FEATURES`. Extractors with heavy
dependencies import them lazily, so registration itself is always cheap.
"""

from tulip.features.registries import AUDIO_FEATURES, TEXT_FEATURES


def _register_builtins() -> None:
    from tulip.features import audio, text  # noqa: F401


_register_builtins()

__all__ = ["AUDIO_FEATURES", "TEXT_FEATURES"]
