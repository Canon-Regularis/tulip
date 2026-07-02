"""Linear-model explanations read directly from TF-IDF pipeline coefficients.

For a fitted :class:`sklearn.pipeline.Pipeline` whose final step is a linear
classifier (anything exposing ``coef_``), the contribution of feature ``j`` to
the score of class ``c`` for one input is exactly ``x[j] * coef_[c, j]``. This
is the fastest and most faithful explanation available for the classical
TF-IDF baselines: no sampling, no surrogate model.

``CalibratedClassifierCV`` (used by tulip's ``linear_svm``) hides the linear
model behind per-fold calibrators; see :func:`_linear_coefficients` for the
averaging approximation applied in that case.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.pipeline import FeatureUnion

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import Explanation, TokenAttribution
from tulip.explain._shared import as_text
from tulip.explain.registry import EXPLAINERS
from tulip.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["TopTfidfExplainer", "class_top_features"]


def _split_pipeline(pipeline: Any) -> tuple[Any, Any]:
    """Split a fitted sklearn Pipeline into (transformer part, final estimator).

    Args:
        pipeline: A fitted :class:`sklearn.pipeline.Pipeline`.

    Returns:
        ``(transformer, estimator)`` where ``transformer`` maps raw documents
        to the feature matrix the estimator was trained on.

    Raises:
        ConfigurationError: if ``pipeline`` is not Pipeline-like.
    """
    steps = getattr(pipeline, "steps", None)
    if not steps:
        raise ConfigurationError(
            "top_tfidf requires a fitted sklearn Pipeline (vectorizer + linear model); "
            f"got {type(pipeline).__name__}"
        )
    return pipeline[:-1], steps[-1][1]


def _linear_coefficients(estimator: Any) -> tuple[np.ndarray, bool]:
    """Extract a ``(n_classes, n_features)`` coefficient matrix from ``estimator``.

    For :class:`sklearn.calibration.CalibratedClassifierCV` the coefficients of
    the underlying linear estimators (one per calibration fold) are averaged.
    This is an approximation: calibration applies a monotone (sigmoid or
    isotonic) map per class, so the averaged raw-margin coefficients preserve
    the *direction* and relative ordering of feature evidence but not the
    calibrated probability scale. That trade-off is acceptable for "which
    words pushed this prediction" reports.

    Args:
        estimator: The fitted final step of a classification pipeline.

    Returns:
        ``(coefficients, calibrated)`` where ``calibrated`` flags the
        averaging approximation described above.

    Raises:
        ConfigurationError: if the estimator exposes no linear coefficients.
    """
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        matrix = np.asarray(coef.toarray() if sparse.issparse(coef) else coef, dtype=np.float64)
        return np.atleast_2d(matrix), False

    folds = getattr(estimator, "calibrated_classifiers_", None)
    if folds:
        matrices: list[np.ndarray] = []
        for fold in folds:
            inner = getattr(fold, "estimator", None)
            if inner is None:  # sklearn < 1.2 spelling
                inner = getattr(fold, "base_estimator", None)
            inner_coef = getattr(inner, "coef_", None)
            if inner_coef is None:
                break
            dense = inner_coef.toarray() if sparse.issparse(inner_coef) else inner_coef
            matrices.append(np.atleast_2d(np.asarray(dense, dtype=np.float64)))
        if matrices and len(matrices) == len(folds):
            return np.mean(matrices, axis=0), True

    raise ConfigurationError(
        f"top_tfidf requires a linear classifier exposing coef_ (or a CalibratedClassifierCV "
        f"wrapping one); {type(estimator).__name__} does not. For non-linear models use the "
        f"'lime' or 'shap' explainer instead."
    )


def _class_rows(coefficients: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Expand binary ``(1, n_features)`` coefficients to one row per class.

    sklearn stores a single row for binary problems, oriented towards
    ``classes_[1]``; the row for ``classes_[0]`` is its negation.

    Args:
        coefficients: Matrix of shape ``(n_classes, n_features)`` or
            ``(1, n_features)`` for binary models.
        classes: The estimator's ``classes_`` array.

    Returns:
        Matrix of shape ``(len(classes), n_features)``.

    Raises:
        ConfigurationError: if the shapes cannot be reconciled.
    """
    if coefficients.shape[0] == len(classes):
        return coefficients
    if coefficients.shape[0] == 1 and len(classes) == 2:
        return np.vstack([-coefficients[0], coefficients[0]])
    raise ConfigurationError(
        f"coefficient matrix has {coefficients.shape[0]} rows but the model has "
        f"{len(classes)} classes"
    )


def _union_prefixes(transformer: Any) -> set[str]:
    """Collect FeatureUnion branch names anywhere inside ``transformer``.

    Used to decide which ``prefix__feature`` names are genuine FeatureUnion
    prefixes (to be rewritten as ``"prefix: feature"``) rather than literal
    double underscores inside a token.
    """
    prefixes: set[str] = set()
    stack: list[Any] = [transformer]
    while stack:
        obj = stack.pop()
        if isinstance(obj, FeatureUnion):
            prefixes.update(name for name, _ in obj.transformer_list)
        steps = getattr(obj, "steps", None)
        if steps:
            stack.extend(step for _, step in steps)
        transformer_list = getattr(obj, "transformer_list", None)
        if transformer_list:
            stack.extend(step for _, step in transformer_list)
    return prefixes


def _feature_names(transformer: Any, n_features: int) -> list[str]:
    """Return readable feature names for the transformed space.

    FeatureUnion prefixes (``word__ale``) are rewritten to a readable
    ``"word: ale"`` form; names from a plain vectorizer pass through untouched.

    Args:
        transformer: The transformer part of the pipeline.
        n_features: Expected number of features (for the fallback).

    Returns:
        One display name per feature column.
    """
    if not hasattr(transformer, "get_feature_names_out"):
        logger.warning(
            "%s has no get_feature_names_out; falling back to positional feature names",
            type(transformer).__name__,
        )
        return [f"feature_{index}" for index in range(n_features)]
    raw_names = [str(name) for name in transformer.get_feature_names_out()]
    prefixes = _union_prefixes(transformer)
    if not prefixes:
        return raw_names
    readable: list[str] = []
    for name in raw_names:
        prefix, separator, feature = name.partition("__")
        if separator and prefix in prefixes:
            readable.append(f"{prefix}: {feature}")
        else:
            readable.append(name)
    return readable


@EXPLAINERS.register("top_tfidf")
class TopTfidfExplainer:
    """Signed per-feature contributions for one input under a linear pipeline.

    For the predicted class, each feature contributes exactly
    ``feature_value * class_coefficient`` to the decision score, so the
    returned attributions are an exact decomposition of the (uncalibrated)
    linear score — positive weights pushed towards the predicted class,
    negative weights pushed away from it.

    Attributes:
        top_k: How many positive and how many negative contributions to keep
            (up to ``2 * top_k`` attributions in total).
    """

    def __init__(self, top_k: int = 10) -> None:
        """Configure the explainer.

        Args:
            top_k: Number of strongest positive and strongest negative
                contributions to include.

        Raises:
            ConfigurationError: if ``top_k`` is not positive.
        """
        if top_k < 1:
            raise ConfigurationError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k

    def explain(self, pipeline: Any, raw_input: Any, **kwargs: Any) -> Explanation:
        """Explain one prediction from the pipeline's linear coefficients.

        Args:
            pipeline: A fitted sklearn Pipeline whose final step is a linear
                classifier (``coef_``) or a ``CalibratedClassifierCV`` around
                one.
            raw_input: The raw document to explain.
            **kwargs: ``top_k`` overrides the constructor value.

        Returns:
            An :class:`Explanation` with signed :class:`TokenAttribution`
            entries (strongest positive first, then strongest negative).

        Raises:
            ConfigurationError: if the pipeline is not a linear-model Pipeline.
        """
        top_k = int(kwargs.get("top_k", self.top_k))
        text = as_text(raw_input)
        transformer, estimator = _split_pipeline(pipeline)
        coefficients, calibrated = _linear_coefficients(estimator)
        classes = np.asarray(estimator.classes_)
        rows = _class_rows(coefficients, classes)

        predicted_label = str(pipeline.predict([text])[0])
        class_index = int(np.flatnonzero(classes.astype(str) == predicted_label)[0])
        class_row = rows[class_index]

        features = transformer.transform([text])
        if sparse.issparse(features):
            row = sparse.csr_matrix(features)
            indices = row.indices
            contributions = row.data * class_row[indices]
        else:
            dense = np.asarray(features, dtype=np.float64)[0]
            indices = np.flatnonzero(dense)
            contributions = dense[indices] * class_row[indices]

        names = _feature_names(transformer, rows.shape[1])
        order = np.argsort(contributions)
        negative = [pos for pos in order[:top_k] if contributions[pos] < 0]
        positive = [pos for pos in order[::-1][:top_k] if contributions[pos] > 0]
        attributions = tuple(
            TokenAttribution(token=names[indices[pos]], weight=float(contributions[pos]))
            for pos in [*positive, *negative]
        )
        return Explanation(
            method="top_tfidf",
            predicted_label=predicted_label,
            attributions=attributions,
            details={
                "classes": [str(label) for label in classes],
                "calibrated_average": calibrated,
                "active_features": len(indices),
            },
        )


def class_top_features(pipeline: Any, k: int = 10) -> dict[str, tuple[TokenAttribution, ...]]:
    """Return the top-k globally most indicative features per class.

    Reads the linear model's coefficient matrix directly (input-independent),
    answering "which words indicate each dialect" for report tables. For
    ``CalibratedClassifierCV`` the per-fold coefficients are averaged (see
    :func:`_linear_coefficients` for the approximation involved).

    Args:
        pipeline: A fitted sklearn Pipeline ending in a linear classifier.
        k: Number of features to return per class.

    Returns:
        Mapping from class label to its ``k`` highest-coefficient features as
        :class:`TokenAttribution` entries, strongest first.

    Raises:
        ConfigurationError: if ``k < 1`` or the pipeline is not linear.
    """
    if k < 1:
        raise ConfigurationError(f"k must be >= 1, got {k}")
    transformer, estimator = _split_pipeline(pipeline)
    coefficients, _ = _linear_coefficients(estimator)
    classes = np.asarray(estimator.classes_)
    rows = _class_rows(coefficients, classes)
    names = _feature_names(transformer, rows.shape[1])

    result: dict[str, tuple[TokenAttribution, ...]] = {}
    for class_index, label in enumerate(classes):
        row = rows[class_index]
        top = np.argsort(row)[::-1][:k]
        result[str(label)] = tuple(
            TokenAttribution(token=names[j], weight=float(row[j])) for j in top
        )
    return result
