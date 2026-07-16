"""Fitted-state checks, hyperparameter validation, and the argmax-predict mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tulip.core.exceptions import ConfigurationError, TulipError

if TYPE_CHECKING:
    from collections.abc import Sequence


def require_fitted(estimator: Any, *attributes: str) -> None:
    """Raise if ``estimator`` has not been fitted yet.

    Args:
        estimator: The wrapper instance to check.
        attributes: Post-fit attribute names that must be present.

    Raises:
        TulipError: if any required attribute is missing.
    """
    for attribute in attributes:
        if getattr(estimator, attribute, None) is None:
            raise TulipError(
                f"{type(estimator).__name__} is not fitted yet; call fit(X, y) before predicting"
            )


def validate_class_weight(estimator: Any) -> None:
    """Validate the shared ``class_weight`` knob (``None`` or ``"balanced"``).

    Split out so wrappers that do not share the full training-param set (the
    embedding speech head) can run the identical check on their own.

    Raises:
        ConfigurationError: if ``class_weight`` is neither ``None`` nor
            ``"balanced"``.
    """
    if estimator.class_weight not in (None, "balanced"):
        raise ConfigurationError(
            f'class_weight must be None or "balanced", got {estimator.class_weight!r}'
        )


def validate_common_training_params(estimator: Any) -> None:
    """Validate the training knobs shared by every fine-tuning wrapper.

    Called from each wrapper's ``fit`` (never ``__init__``: the sklearn
    estimator contract requires ``set_params``-injected values to be
    validated too). Wrappers add their modality-specific checks on top.

    Raises:
        ConfigurationError: if a hyperparameter is out of range.
    """
    if estimator.epochs < 1:
        raise ConfigurationError(f"epochs must be >= 1, got {estimator.epochs}")
    if estimator.batch_size < 1:
        raise ConfigurationError(f"batch_size must be >= 1, got {estimator.batch_size}")
    if estimator.learning_rate <= 0:
        raise ConfigurationError(f"learning_rate must be > 0, got {estimator.learning_rate}")
    if not 0.0 <= estimator.warmup_ratio <= 1.0:
        raise ConfigurationError(f"warmup_ratio must be in [0, 1], got {estimator.warmup_ratio}")
    if estimator.gradient_accumulation_steps < 1:
        raise ConfigurationError(
            f"gradient_accumulation_steps must be >= 1, got {estimator.gradient_accumulation_steps}"
        )
    validate_class_weight(estimator)


class ArgmaxPredictMixin:
    """Derive ``predict`` from ``predict_proba`` (argmax over ``classes_``).

    Guarantees ``predict == classes_[argmax(predict_proba)]`` by construction
    for every wrapper that mixes this in, instead of four hand-maintained
    copies of the same line.
    """

    classes_: np.ndarray

    if TYPE_CHECKING:  # concrete classes provide the real implementation

        def predict_proba(self, X: Sequence[Any]) -> np.ndarray: ...

    def predict(self, X: Sequence[Any]) -> np.ndarray:
        """Return the most probable class label for each input."""
        probabilities = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(probabilities, axis=1)]


# ---------------------------------------------------------------- torch loops
