"""Model zoo: classical baselines, transformer text models, and speech models.

Importing this package registers all built-in models in :data:`MODELS`.
Models with heavy dependencies (torch, transformers, speechbrain, fasttext)
import them lazily inside methods, so registration itself is always cheap.
"""

from tulip.models.registry import MODELS


def _register_builtins() -> None:
    from tulip.models import (  # noqa: F401
        classical,
        ensemble,
        fasttext_model,
        neural_audio,
        neural_text,
    )


_register_builtins()

# The documented public surface, importable from the subsystem root like
# every sibling package (the registration imports above make these free).
from tulip.models.calibration import (  # noqa: E402
    IdentityCalibrator,
    IsotonicCalibrator,
    ProbabilityCalibrator,
    TemperatureScaling,
)
from tulip.models.fasttext_model import FastTextClassifier  # noqa: E402
from tulip.models.neural_audio import (  # noqa: E402
    EmbeddingSpeechClassifier,
    FinetunedSpeechClassifier,
    SpeechClassifier,
)
from tulip.models.neural_text import TransformerTextClassifier  # noqa: E402
from tulip.models.persistence import load_model, save_model  # noqa: E402

__all__ = [
    "MODELS",
    "EmbeddingSpeechClassifier",
    "FastTextClassifier",
    "FinetunedSpeechClassifier",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "ProbabilityCalibrator",
    "SpeechClassifier",
    "TemperatureScaling",
    "TransformerTextClassifier",
    "load_model",
    "save_model",
]
