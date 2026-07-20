"""Classical baseline classifiers for dialect classification.

Registers factory functions in :data:`tulip.models.MODELS` under the canonical
names ``naive_bayes``, ``logistic_regression``, ``linear_svm``,
``random_forest``, ``xgboost``, and ``lightgbm``. Every factory accepts keyword
parameters that override sensible dialect-classification defaults and returns
an object satisfying :class:`tulip.core.interfaces.Classifier`
(``fit``/``predict``/``predict_proba``/``classes_``).

All factories accept ``random_state`` (or its alias ``seed``); estimators
without a stochastic component ignore it. scikit-learn estimators are imported
inside the factories so that importing this module, and therefore registering
the models, stays cheap; the gradient-boosting factories additionally guard
their optional dependencies via :func:`tulip.utils.optional.optional_import`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# sklearn.base is imported at module level (unlike the factory estimators below,
# which stay lazy) because LabelEncodedClassifier must subclass these to be a
# real estimator: without the sklearn tags they supply, scikit-learn >= 1.6
# raises AttributeError when it resolves tags on the wrapper during predict or
# inside voting/stacking ensembles.
from sklearn.base import BaseEstimator, ClassifierMixin

from tulip.core.exceptions import ConfigurationError
from tulip.models._factory import pop_seed
from tulip.models.registry import MODELS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

logger = get_logger(__name__)

#: Default seed used when neither ``random_state`` nor ``seed`` is supplied,
#: so every baseline is reproducible out of the box.
DEFAULT_SEED = 0

__all__ = [
    "DEFAULT_SEED",
    "LabelEncodedClassifier",
    "lightgbm",
    "linear_svm",
    "logistic_regression",
    "majority",
    "naive_bayes",
    "random_forest",
    "xgboost",
]


def _pop_seed(params: dict[str, Any], *, default: int | None = DEFAULT_SEED) -> int | None:
    """Extract the random seed from ``params``, honouring both spellings.

    A thin wrapper over the shared :func:`tulip.models._factory.pop_seed` that
    only supplies this module's default; the reconciliation and its conflict
    check live in one place there.
    """
    return pop_seed(params, default=default)


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Adapter exposing string class labels over an integer-label estimator.

    XGBoost rejects non-integer class labels, and LightGBM's handling of them
    varies across versions; tulip pipelines, however, always work with string
    dialect labels. This wrapper encodes ``y`` with a
    :class:`sklearn.preprocessing.LabelEncoder` at fit time and decodes
    predictions on the way out, so ``classes_`` and ``predict`` always speak
    the caller's original label vocabulary. ``predict_proba`` columns follow
    ``classes_`` order (the encoder sorts labels, matching the wrapped
    estimator's encoded-class order).
    """

    def __init__(self, estimator: Any) -> None:
        """Wrap ``estimator``, an object with sklearn-style fit/predict methods."""
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> LabelEncodedClassifier:
        """Fit the wrapped estimator on integer-encoded labels."""
        from sklearn.preprocessing import LabelEncoder

        self._label_encoder = LabelEncoder()
        encoded = self._label_encoder.fit_transform(np.asarray(y))
        self.estimator.fit(X, encoded)
        self.classes_: np.ndarray = self._label_encoder.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict and decode back to the original label vocabulary."""
        encoded = self.estimator.predict(X)
        return self._label_encoder.inverse_transform(np.asarray(encoded, dtype=int))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities with columns ordered like ``classes_``."""
        return self.estimator.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return parameters in scikit-learn's nested ``estimator__*`` style."""
        out: dict[str, Any] = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for key, value in self.estimator.get_params(deep=True).items():
                out[f"estimator__{key}"] = value
        return out

    def set_params(self, **params: Any) -> LabelEncodedClassifier:
        """Set ``estimator`` or nested ``estimator__*`` parameters."""
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        nested = {}
        for key, value in params.items():
            prefix, _, rest = key.partition("__")
            if prefix != "estimator" or not rest:
                raise ConfigurationError(f"unknown parameter {key!r} for LabelEncodedClassifier")
            nested[rest] = value
        if nested:
            self.estimator.set_params(**nested)
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(estimator={self.estimator!r})"


@MODELS.register("majority")
def majority(**params: Any) -> Any:
    """Majority-class baseline: always predicts the most frequent training label.

    The floor every real model must beat. It ignores the features entirely and
    predicts the single most frequent class; ``predict_proba`` returns the class
    priors. On an imbalanced dialect corpus a respectable-looking accuracy can be
    almost entirely this baseline, so it belongs at the top of every benchmark
    table as the reference point.

    Args:
        **params: Overrides forwarded to ``DummyClassifier``. ``random_state``/
            ``seed`` are accepted for interface uniformity but ignored (the
            ``prior`` strategy is deterministic).

    Returns:
        An unfitted ``DummyClassifier(strategy="prior")``.
    """
    from sklearn.dummy import DummyClassifier

    seed = _pop_seed(params, default=None)
    if seed is not None:
        logger.debug("majority baseline is deterministic; ignoring seed %r", seed)
    merged: dict[str, Any] = {"strategy": "prior", **params}
    return DummyClassifier(**merged)


@MODELS.register("naive_bayes")
def naive_bayes(**params: Any) -> Any:
    """Multinomial naive Bayes baseline (default ``alpha=0.1``).

    Works well on TF-IDF term matrices. Caveat: :class:`MultinomialNB`
    requires non-negative features: it is suitable for counts and TF-IDF but
    not for standardised/dense feature sets containing negative values (e.g.
    z-scored stylometry or neural embeddings).

    Args:
        **params: Overrides forwarded to ``MultinomialNB``. ``random_state``/
            ``seed`` are accepted for interface uniformity but ignored (the
            estimator has no stochastic component).

    Returns:
        An unfitted ``MultinomialNB``.
    """
    from sklearn.naive_bayes import MultinomialNB

    seed = _pop_seed(params, default=None)
    if seed is not None:
        logger.debug("naive_bayes has no stochastic component; ignoring seed %r", seed)
    merged: dict[str, Any] = {"alpha": 0.1, **params}
    return MultinomialNB(**merged)


@MODELS.register("logistic_regression")
def logistic_regression(**params: Any) -> Any:
    """Multinomial logistic regression baseline.

    Defaults: ``max_iter=2000`` (high-dimensional sparse TF-IDF converges
    slowly), ``C=1.0``, ``class_weight="balanced"`` (dialect corpora are
    imbalanced), and the ``lbfgs`` solver, which handles sparse input.

    Args:
        **params: Overrides forwarded to ``LogisticRegression``; accepts
            ``random_state``/``seed``.

    Returns:
        An unfitted ``LogisticRegression``.
    """
    from sklearn.linear_model import LogisticRegression

    seed = _pop_seed(params)
    merged: dict[str, Any] = {
        "max_iter": 2000,
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "lbfgs",
        **params,
    }
    return LogisticRegression(random_state=seed, **merged)


@MODELS.register("linear_svm")
def linear_svm(*, method: str = "sigmoid", cv: int = 3, **params: Any) -> Any:
    """Linear SVM baseline with probability calibration.

    ``LinearSVC`` (hinge loss) has no native ``predict_proba``, so it is
    wrapped in :class:`CalibratedClassifierCV` (sigmoid/Platt scaling by
    default) to satisfy the :class:`~tulip.core.interfaces.Classifier`
    contract of always exposing calibrated probabilities.

    Args:
        method: Calibration method (``"sigmoid"`` or ``"isotonic"``).
        cv: Number of calibration cross-validation folds.
        **params: Overrides forwarded to the inner ``LinearSVC`` (default
            ``class_weight="balanced"``); accepts ``random_state``/``seed``.

    Returns:
        An unfitted ``CalibratedClassifierCV`` wrapping a ``LinearSVC``.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    seed = _pop_seed(params)
    merged: dict[str, Any] = {"class_weight": "balanced", "dual": "auto", **params}
    svm = LinearSVC(random_state=seed, **merged)
    return CalibratedClassifierCV(svm, method=method, cv=cv)


@MODELS.register("random_forest")
def random_forest(**params: Any) -> Any:
    """Random forest baseline.

    Defaults: ``n_estimators=300``, ``class_weight="balanced"``, and
    ``n_jobs=-1`` (trees train in parallel).

    Args:
        **params: Overrides forwarded to ``RandomForestClassifier``; accepts
            ``random_state``/``seed``.

    Returns:
        An unfitted ``RandomForestClassifier``.
    """
    from sklearn.ensemble import RandomForestClassifier

    seed = _pop_seed(params)
    merged: dict[str, Any] = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "n_jobs": -1,
        **params,
    }
    return RandomForestClassifier(random_state=seed, **merged)


@MODELS.register("xgboost", metadata={"extra": "boosting"})
def xgboost(**params: Any) -> Any:
    """XGBoost gradient-boosting baseline (optional extra ``boosting``).

    Defaults: ``n_estimators=300``, ``tree_method="hist"`` (fast histogram
    algorithm), ``eval_metric="mlogloss"``, ``n_jobs=-1``. Wrapped in
    :class:`LabelEncodedClassifier` so string dialect labels round-trip and
    ``classes_`` exposes the original label vocabulary.

    Args:
        **params: Overrides forwarded to ``xgboost.XGBClassifier``; accepts
            ``random_state``/``seed``.

    Returns:
        An unfitted :class:`LabelEncodedClassifier` around an ``XGBClassifier``.

    Raises:
        MissingDependencyError: If xgboost is not installed.
    """
    xgb = optional_import(
        "xgboost", extra="boosting", purpose="the xgboost gradient-boosting baseline"
    )
    seed = _pop_seed(params)
    merged: dict[str, Any] = {
        "n_estimators": 300,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        **params,
    }
    return LabelEncodedClassifier(xgb.XGBClassifier(random_state=seed, **merged))


@MODELS.register("lightgbm", metadata={"extra": "boosting"})
def lightgbm(**params: Any) -> Any:
    """LightGBM gradient-boosting baseline (optional extra ``boosting``).

    Defaults: ``n_estimators=300``, ``verbose=-1`` (LightGBM is chatty by
    default), ``n_jobs=-1``. Wrapped in :class:`LabelEncodedClassifier` for
    version-stable handling of string dialect labels.

    Args:
        **params: Overrides forwarded to ``lightgbm.LGBMClassifier``; accepts
            ``random_state``/``seed``.

    Returns:
        An unfitted :class:`LabelEncodedClassifier` around an ``LGBMClassifier``.

    Raises:
        MissingDependencyError: If lightgbm is not installed.
    """
    lgbm = optional_import(
        "lightgbm", extra="boosting", purpose="the lightgbm gradient-boosting baseline"
    )
    seed = _pop_seed(params)
    merged: dict[str, Any] = {
        "n_estimators": 300,
        "verbose": -1,
        "n_jobs": -1,
        **params,
    }
    return LabelEncodedClassifier(lgbm.LGBMClassifier(random_state=seed, **merged))
