"""Late-fusion of a text and an audio classifier over one :class:`Sample` stream.

The package keeps two concerns with different reasons to change apart:

* :mod:`~tulip.pipeline.fusion.strategies`: the opinion-pooling strategy
  family (protocol, weight validation, the three concrete strategies, and the
  name-keyed factory). A leaf module: numpy + :mod:`tulip.core` only.
* :mod:`~tulip.pipeline.fusion.classifier`: :class:`MultimodalClassifier`,
  which composes two bases and persists them.

Both were one 600-line module; splitting them lets the strategies be unit-tested
on hand-built numpy stacks without importing the classifier stack. The public
surface is unchanged: every name below still imports from
``tulip.pipeline.fusion``. ``ProbabilisticClassifier`` now lives in
:mod:`tulip.pipeline.protocols` (it is a general pipeline abstraction, not
fusion's alone) and is re-exported here for backward compatibility.
"""

from __future__ import annotations

from tulip.pipeline.fusion.classifier import MultimodalClassifier
from tulip.pipeline.fusion.comparison import (
    ModalityComparison,
    ModalityScore,
    compare_modalities,
)
from tulip.pipeline.fusion.strategies import (
    ConfidenceWeightedFusion,
    FusionStrategy,
    LogarithmicPoolingFusion,
    MaximumFusion,
    WeightedAverageFusion,
    build_strategy,
    default_params,
)
from tulip.pipeline.protocols import ProbabilisticClassifier

__all__ = [
    "ConfidenceWeightedFusion",
    "FusionStrategy",
    "LogarithmicPoolingFusion",
    "MaximumFusion",
    "ModalityComparison",
    "ModalityScore",
    "MultimodalClassifier",
    "ProbabilisticClassifier",
    "WeightedAverageFusion",
    "build_strategy",
    "compare_modalities",
    "default_params",
]
