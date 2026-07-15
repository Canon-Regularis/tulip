"""Nearest-training-example explanations via cosine similarity.

Example-based evidence complements token attributions: showing the most
similar training utterances (and their gold labels) lets a user judge whether
a prediction rests on genuinely similar dialectal material or on spurious
matches. The index is built once over the training samples and reused for
every query; similarity search is pure numpy/scipy and keeps sparse feature
matrices sparse throughout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import Explanation, NeighborExample, Sample
from tulip.explain._shared import as_text, predicted_label_or_none
from tulip.explain.registry import EXPLAINERS
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = get_logger(__name__)

__all__ = ["NearestExamplesExplainer"]


def _default_label(sample: Sample) -> str | None:
    """Pick the most specific-but-common label level available on a sample."""
    labels = sample.labels
    return labels.dialect or labels.family or labels.region or labels.voivodeship


def _l2_normalize(matrix: Any) -> Any:
    """Row-normalise a dense or sparse matrix to unit L2 norm.

    Zero rows are left as zero vectors (their similarity to anything is 0).
    Sparse input stays sparse: norms are computed from the data array and
    applied via a diagonal multiply, never densifying the matrix.

    Args:
        matrix: 2-D ``numpy.ndarray`` or ``scipy.sparse`` matrix.

    Returns:
        The normalised matrix (CSR when the input was sparse).
    """
    if sparse.issparse(matrix):
        csr = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
        norms = np.sqrt(np.asarray(csr.multiply(csr).sum(axis=1)).ravel())
        scale = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        return sparse.diags(scale) @ csr
    dense = np.asarray(matrix, dtype=np.float64)
    if dense.ndim != 2:
        raise ConfigurationError(f"expected a 2-D feature matrix, got shape {dense.shape}")
    norms = np.linalg.norm(dense, axis=1)
    scale = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    return dense * scale[:, np.newaxis]


@EXPLAINERS.register("nearest_examples")
class NearestExamplesExplainer:
    """Retrieve the most cosine-similar training samples as evidence.

    Usage: call :meth:`index` once with the training samples and the (fitted)
    feature transformer used by the classifier, then call :meth:`explain` per
    query. The transformer must accept a sequence of raw texts and return one
    feature row per text (any tulip text feature extractor or the transformer
    part of a fitted sklearn Pipeline qualifies).

    Attributes:
        k: Default number of neighbours to return.
        snippet_chars: Maximum characters of neighbour text to include.
    """

    def __init__(
        self,
        k: int = 5,
        *,
        snippet_chars: int = 200,
        label_of: Callable[[Sample], str | None] | None = None,
    ) -> None:
        """Configure the explainer.

        Args:
            k: Default number of neighbours returned per query.
            snippet_chars: Truncation length for neighbour text snippets.
            label_of: Optional override selecting which gold label to show per
                sample; defaults to dialect, falling back through family,
                region, and voivodeship.

        Raises:
            ConfigurationError: if ``k`` or ``snippet_chars`` is not positive.
        """
        if k < 1:
            raise ConfigurationError(f"k must be >= 1, got {k}")
        if snippet_chars < 1:
            raise ConfigurationError(f"snippet_chars must be >= 1, got {snippet_chars}")
        self.k = k
        self.snippet_chars = snippet_chars
        self._label_of = label_of or _default_label
        self._matrix: Any = None
        self._transformer: Any = None
        self._ids: list[str] = []
        self._labels: list[str | None] = []
        self._texts: list[str] = []

    def index(self, samples: Sequence[Sample], transformer: Any) -> NearestExamplesExplainer:
        """Build the similarity index over the training samples.

        Args:
            samples: Training samples (their ``text`` is embedded; samples
                without text contribute zero vectors and are never retrieved).
            transformer: A fitted feature extractor with
                ``transform(texts) -> matrix`` producing one row per text.

        Returns:
            ``self``, ready for :meth:`explain`.

        Raises:
            ConfigurationError: if ``samples`` is empty or the transformer
                output does not align with the samples.
        """
        if not samples:
            raise ConfigurationError("cannot build a nearest-examples index from zero samples")
        if not hasattr(transformer, "transform"):
            raise ConfigurationError(
                f"transformer must expose transform(texts); got {type(transformer).__name__}"
            )
        texts = [sample.text or "" for sample in samples]
        matrix = transformer.transform(texts)
        n_rows = matrix.shape[0]
        if n_rows != len(samples):
            raise ConfigurationError(
                f"transformer produced {n_rows} rows for {len(samples)} samples"
            )
        self._matrix = _l2_normalize(matrix)
        self._transformer = transformer
        self._ids = [sample.id for sample in samples]
        self._labels = [self._label_of(sample) for sample in samples]
        self._texts = texts
        logger.debug("indexed %d samples for nearest-example retrieval", len(samples))
        return self

    def explain(self, pipeline: Any, raw_input: Any, **kwargs: Any) -> Explanation:
        """Return the top-k most similar indexed samples for ``raw_input``.

        Args:
            pipeline: Optional fitted classifier used only to fill
                ``predicted_label``; retrieval itself relies solely on the
                index. May be ``None``.
            raw_input: The raw query document.
            **kwargs: ``k`` and ``snippet_chars`` override the constructor
                values.

        Returns:
            An :class:`Explanation` whose ``neighbors`` are sorted by
            descending cosine similarity.

        Raises:
            ConfigurationError: if :meth:`index` has not been called.
        """
        if self._matrix is None:
            raise ConfigurationError(
                "nearest_examples has no index; call index(samples, transformer) first"
            )
        k = int(kwargs.get("k", self.k))
        if k < 1:
            raise ConfigurationError(f"k must be >= 1, got {k}")
        snippet_chars = int(kwargs.get("snippet_chars", self.snippet_chars))
        text = as_text(raw_input)

        query = _l2_normalize(self._transformer.transform([text]))
        product = self._matrix @ query.T  # (n_indexed, 1); only this densifies
        similarities = (
            product.toarray().ravel() if sparse.issparse(product) else np.asarray(product).ravel()
        )
        k = min(k, len(similarities))
        top = np.argsort(similarities)[::-1][:k]

        neighbors = tuple(
            NeighborExample(
                sample_id=self._ids[i],
                label=self._labels[i],
                text=self._snippet(self._texts[i], snippet_chars),
                similarity=float(np.clip(similarities[i], -1.0, 1.0)),
            )
            for i in top
        )
        predicted_label = predicted_label_or_none(pipeline, text)
        return Explanation(
            method="nearest_examples",
            predicted_label=predicted_label,
            neighbors=neighbors,
            details={"index_size": len(self._ids), "k": k},
        )

    def _snippet(self, text: str, limit: int) -> str:
        """Truncate neighbour text for display."""
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."
