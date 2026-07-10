"""Shared scaffolding for the dense (non-vectorizer) text feature extractors.

:class:`~tulip.features.text.keywords.DialectKeywordExtractor`,
:class:`~tulip.features.text.phonology.PhonologicalMarkerExtractor`, and
:class:`~tulip.features.text.affixes.AffixFrequencyExtractor` each emit a small,
named, dense feature block and repeated the same two members verbatim: a
not-fitted guard and a ``get_feature_names_out`` that returns
``feature_names_``. :class:`_DenseTextExtractor` provides both once.

**Why the base defines no ``__init__``.** scikit-learn's ``get_params`` /
``clone`` introspect the *concrete* class's ``__init__`` signature to discover a
transformer's hyper-parameters. A base ``__init__`` would shadow that and quietly
break ``clone()`` and ``GridSearchCV``. Each concrete extractor therefore keeps
its own ``__init__`` with its own named parameters; the base contributes only
behaviour that does not touch construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DenseTextExtractor"]


class DenseTextExtractor(TransformerMixin, BaseEstimator):
    """Base for dense, named text feature extractors (fitted-state + feature names).

    Subclasses set their fitted attributes -- always including
    ``feature_names_`` -- in ``fit``, implement ``transform``, and (unless they
    guard on a different attribute) inherit the fitted check and
    ``get_feature_names_out`` unchanged. The public name is exported without the
    leading underscore for subclasses in sibling modules; it is not part of the
    package's public API.
    """

    #: Attribute whose presence marks the estimator as fitted. Subclasses that
    #: set ``feature_names_`` last may leave this; those that gate on a different
    #: attribute (e.g. a learned vocabulary) override it.
    _fitted_attr: ClassVar[str] = "feature_names_"

    #: Populated by ``fit``; the dense block's column names, in order.
    feature_names_: tuple[str, ...]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the fitted feature column names as an object array."""
        self._check_fitted()
        return np.asarray(self.feature_names_, dtype=object)

    def _check_fitted(self) -> None:
        """Raise :class:`NotFittedError` if ``fit`` has not run.

        Raises:
            NotFittedError: if the estimator has not been fitted.
        """
        if not hasattr(self, self._fitted_attr):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet; call fit first."
            )

    # transform is subclass responsibility; declared for the type checker only.
    if TYPE_CHECKING:

        def transform(self, X: Sequence[str]) -> np.ndarray: ...
