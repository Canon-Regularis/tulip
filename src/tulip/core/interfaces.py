"""Structural interfaces implemented across tulip subsystems.

Feature extractors and classifiers follow scikit-learn conventions
(``fit``/``transform``/``predict``/``predict_proba``) so that components
compose freely with :class:`sklearn.pipeline.Pipeline` and
:class:`sklearn.pipeline.FeatureUnion` regardless of whether they are backed
by scikit-learn, gradient boosting, or neural networks.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from tulip.core.exceptions import DataError

if TYPE_CHECKING:
    import numpy as np

    from tulip.core.types import DatasetInfo, Explanation, Sample


@runtime_checkable
class FeatureExtractor(Protocol):
    """Transforms raw inputs (texts or audio paths) into feature matrices.

    ``X`` is a sequence of raw documents for text features, or a sequence of
    audio file paths for audio features. ``transform`` returns a 2-D array or
    sparse matrix with one row per input.
    """

    def fit(self, X: Sequence[Any], y: Any = None) -> FeatureExtractor: ...

    def transform(self, X: Sequence[Any]) -> Any: ...


@runtime_checkable
class Classifier(Protocol):
    """A trainable multiclass classifier with calibrated probability output.

    Implementations must expose ``classes_`` after fitting. Models without
    native probabilities (e.g. hinge-loss SVMs) must wrap themselves in a
    calibration layer so ``predict_proba`` is always available.

    Label contract: the pipeline layer always passes labels as strings
    (:meth:`DialectClassifier.labelled_batch` stringifies them), so wrappers
    may coerce ``y`` to ``str`` and expose string ``classes_``; code comparing
    predictions against gold labels should compare as strings.
    """

    classes_: np.ndarray

    def fit(self, X: Any, y: Any) -> Classifier: ...

    def predict(self, X: Any) -> np.ndarray: ...

    def predict_proba(self, X: Any) -> np.ndarray: ...


@runtime_checkable
class Explainer(Protocol):
    """Produces an :class:`~tulip.core.types.Explanation` for one input."""

    def explain(self, pipeline: Any, raw_input: Any, **kwargs: Any) -> Explanation: ...


class DatasetLoader(abc.ABC):
    """Loads one source corpus into the canonical :class:`Sample` stream.

    Loaders read from a local directory (``root``) whose expected layout is
    documented per dataset in ``docs/datasets.md``; tulip never scrapes web
    pages at runtime. Corpora with a licence-clean, stable bulk source may
    additionally implement :meth:`download` (set ``auto_downloadable`` and
    override); everything else documents its manual steps in
    :attr:`acquisition`, which ``tulip data download`` surfaces to the user.
    Loaders must be lazy (yield samples) so large corpora never need to fit
    in memory.
    """

    #: Whether :meth:`download` can fetch this corpus without manual steps.
    auto_downloadable: ClassVar[bool] = False

    #: Short human instructions for manual acquisition (the long form lives
    #: in docs/datasets.md). Empty means "see the catalog URL".
    acquisition: ClassVar[str] = ""

    @property
    @abc.abstractmethod
    def info(self) -> DatasetInfo:
        """Static metadata about the corpus (name, URL, tier, licence)."""

    @abc.abstractmethod
    def load(self, root: Path) -> Iterator[Sample]:
        """Yield samples from a local copy of the corpus rooted at ``root``.

        Raises:
            DataError: if the expected files are missing or malformed.
        """

    def download(self, root: Path, **options: Any) -> None:
        """Fetch a local copy of the corpus into ``root``.

        Only meaningful when ``auto_downloadable`` is true; the base
        implementation refuses so callers cannot mistake a manual corpus for
        an automatable one.

        Args:
            root: Directory the corpus should be materialised into.
            **options: Downloader-specific knobs (e.g. ``limit``).

        Raises:
            DataError: always, for corpora without an automatic source.
        """
        del root, options
        raise DataError(
            f"{self.info.name} has no automatic download "
            f"({self.acquisition or f'see {self.info.url} and docs/datasets.md'})"
        )

    def is_available(self, root: Path) -> bool:
        """Whether a local copy of the corpus appears to exist under ``root``."""
        return root.is_dir() and any(root.iterdir())
