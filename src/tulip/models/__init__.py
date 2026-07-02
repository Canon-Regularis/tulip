"""Model zoo: classical baselines, transformer text models, and speech models.

Importing this package registers all built-in models in :data:`MODELS`.
Models with heavy dependencies (torch, transformers, speechbrain, fasttext)
import them lazily inside methods, so registration itself is always cheap.
"""

from tulip.models.registry import MODELS


def _register_builtins() -> None:
    from tulip.models import classical, fasttext_model, neural_audio, neural_text  # noqa: F401


_register_builtins()

__all__ = ["MODELS"]
