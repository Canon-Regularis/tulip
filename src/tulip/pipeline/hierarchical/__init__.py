"""Hierarchical family->dialect backoff classification.

The package keeps its two concerns in separate modules -- the backoff-policy
family (:mod:`.policies`) and the classifier that walks levels and projects onto
families (:mod:`.classifier`) -- while re-exporting the flat public surface the
module presented before it was split, so ``tulip.pipeline.hierarchical`` remains
a single import point.
"""

from __future__ import annotations

from tulip.pipeline.hierarchical.classifier import (
    HierarchicalConfig,
    HierarchicalDialectClassifier,
)
from tulip.pipeline.hierarchical.policies import (
    AllOf,
    AlwaysAccept,
    AnyOf,
    BackoffPolicy,
    ConfidenceThreshold,
    MarginThreshold,
    NotAbstained,
    PolicySpec,
    policy_from_spec,
)

__all__ = [
    "AllOf",
    "AlwaysAccept",
    "AnyOf",
    "BackoffPolicy",
    "ConfidenceThreshold",
    "HierarchicalConfig",
    "HierarchicalDialectClassifier",
    "MarginThreshold",
    "NotAbstained",
    "PolicySpec",
    "policy_from_spec",
]
