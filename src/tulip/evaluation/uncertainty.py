"""Aleatoric versus epistemic uncertainty from ensemble members.

A single probability vector conflates two very different doubts: noise the data
cannot resolve (aleatoric), and the model's own ignorance from too little
training (epistemic). A deep ensemble separates them. If the members agree but
spread their mass, the doubt is aleatoric; if the members disagree, it is
epistemic, and more data would help.

The split is the standard information-theoretic one over the member matrix:

* total (predictive entropy) is the entropy of the mean member probability,
  ``H[E[p]]``;
* aleatoric is the mean of the members' entropies, ``E[H[p]]``;
* epistemic is their difference, the mutual information (BALD), which is
  non-negative.

:func:`decompose_uncertainty` is a pure function over a member-probability array,
so it is deterministic and easy to test. :func:`member_probabilities` extracts
that array from a fitted ensemble (voting or stacking); MC-dropout for the neural
models is out of scope here because it needs a seeded stochastic pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from tulip.core.types import Sample
    from tulip.pipeline.classifier import DialectClassifier

__all__ = [
    "UncertaintyReport",
    "decompose_uncertainty",
    "member_probabilities",
    "uncertainty_report",
]


class UncertaintyReport(BaseModel):
    """Mean total, aleatoric, and epistemic uncertainty over a sample set."""

    model_config = ConfigDict(frozen=True)

    n_samples: int = Field(ge=1)
    n_members: int = Field(ge=2)
    mean_total: float = Field(ge=0.0)
    mean_aleatoric: float = Field(ge=0.0)
    mean_epistemic: float = Field(ge=0.0)

    def to_markdown(self) -> str:
        """Render the mean uncertainty components as a small markdown table."""
        from tulip.evaluation._format import format_metric, markdown_table

        rows = [
            ("total (predictive entropy)", format_metric(self.mean_total)),
            ("aleatoric (data noise)", format_metric(self.mean_aleatoric)),
            ("epistemic (model doubt)", format_metric(self.mean_epistemic)),
        ]
        title = f"# Uncertainty - {self.n_members} members over {self.n_samples} samples"
        return f"{title}\n\n{markdown_table(('Component', 'Mean (nats)'), rows)}"


def _entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats over the last axis, with ``0 log 0 = 0``."""
    proba = np.asarray(proba, dtype=np.float64)
    positive = proba > 0.0
    safe = np.where(positive, proba, 1.0)  # log(1) = 0 avoids a divide-by-zero warning
    return -np.sum(np.where(positive, proba * np.log(safe), 0.0), axis=-1)


def decompose_uncertainty(
    member_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split predictive uncertainty into total, aleatoric, and epistemic.

    Args:
        member_proba: Array of shape ``(n_members, n_samples, n_classes)``.

    Returns:
        Three ``(n_samples,)`` arrays: ``(total, aleatoric, epistemic)``, each in
        nats. Epistemic is the mutual information and is clipped at 0 to absorb
        floating-point error.

    Raises:
        ConfigurationError: if the array is not 3-D or has fewer than two members.
    """
    arr = np.asarray(member_proba, dtype=np.float64)
    if arr.ndim != 3:
        raise ConfigurationError(
            f"member_proba must be (n_members, n_samples, n_classes), got shape {arr.shape}"
        )
    if arr.shape[0] < 2:
        raise ConfigurationError("uncertainty decomposition needs at least two members")
    total = _entropy(arr.mean(axis=0))
    aleatoric = _entropy(arr).mean(axis=0)
    epistemic = np.clip(total - aleatoric, 0.0, None)
    return total, aleatoric, epistemic


def member_probabilities(classifier: DialectClassifier, raws: Sequence[Any]) -> np.ndarray:
    """Extract the per-member probability array from a fitted ensemble classifier.

    Args:
        classifier: A fitted :class:`~tulip.pipeline.classifier.DialectClassifier`
            whose model is a voting or stacking ensemble.
        raws: Raw inputs to score.

    Returns:
        An array of shape ``(n_members, n_samples, n_classes)`` with columns
        aligned to ``classifier.classes_``.

    Raises:
        ConfigurationError: if the classifier is not fitted, or its model exposes
            no member estimators (not an ensemble), or a member lacks
            ``predict_proba``.
    """
    from sklearn.pipeline import Pipeline

    if classifier.pipeline_ is None:
        raise ConfigurationError("classifier is not fitted; call fit(X, y) first")
    pipeline = classifier.pipeline_
    inputs = list(raws)
    if isinstance(pipeline, Pipeline):
        model = pipeline[-1]
        features = pipeline[:-1].transform(inputs)
    else:
        model = pipeline
        features = inputs

    estimators = getattr(model, "estimators_", None)
    if not estimators:
        raise ConfigurationError(
            "uncertainty decomposition needs an ensemble model with member estimators "
            f"(voting or stacking); {type(model).__name__} exposes none"
        )
    n_classes = len(classifier.classes_)
    stack = [_member_proba(member, features, n_classes) for member in estimators]
    return np.stack(stack, axis=0)


def _member_proba(member: Any, features: Any, n_classes: int) -> np.ndarray:
    """One member's probability matrix, its columns already in the ensemble order.

    A voting or stacking ensemble fits every member over the same, consistently
    ordered class set, so each member's ``predict_proba`` columns already align
    to ``classifier.classes_``; only the column count is checked.
    """
    if not hasattr(member, "predict_proba"):
        raise ConfigurationError(
            f"ensemble member {type(member).__name__} has no predict_proba; "
            "uncertainty decomposition needs soft members"
        )
    proba = np.asarray(member.predict_proba(features), dtype=np.float64)
    if proba.shape[1] != n_classes:
        raise ConfigurationError(
            f"ensemble member returned {proba.shape[1]} columns, expected {n_classes}"
        )
    return proba


def uncertainty_report(
    classifier: DialectClassifier, samples: Sequence[Sample]
) -> UncertaintyReport:
    """Decompose uncertainty for samples and average each component.

    Raises:
        ConfigurationError: if the model is not an ensemble (see
            :func:`member_probabilities`).
        DataError: if any sample lacks the classifier's input modality.
    """
    from tulip.pipeline._assembly import raws_for_task

    members = member_probabilities(classifier, raws_for_task(samples, classifier.task))
    total, aleatoric, epistemic = decompose_uncertainty(members)
    return UncertaintyReport(
        n_samples=int(members.shape[1]),
        n_members=int(members.shape[0]),
        mean_total=float(total.mean()),
        mean_aleatoric=float(aleatoric.mean()),
        mean_epistemic=float(epistemic.mean()),
    )
