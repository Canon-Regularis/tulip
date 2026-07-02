"""fastText supervised baseline for dialect text classification.

:class:`FastTextClassifier` wraps ``fasttext`` (the ``fasttext-wheel``
distribution, optional extra ``fasttext``) in the scikit-learn estimator API.
fastText's supervised trainer only reads a plain-text file with one
``__label__<label> <text>`` line per example, so ``fit`` writes a temporary
UTF-8 training file and deletes it after training. Labels are percent-encoded
into the ``__label__`` token so arbitrary label strings (spaces, slashes,
Polish diacritics) survive the round trip; texts are collapsed to single
lines because the format is line-based.

``predict_proba`` asks fastText for the full label distribution
(``k=-1, threshold=0.0``) and maps it back onto the fixed ``classes_`` order.
The label helpers are pure functions so they stay unit-testable without
fastText installed; the fasttext import itself happens lazily inside methods
via :func:`tulip.utils.optional.optional_import`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.models.neural_text import require_fitted
from tulip.models.registry import MODELS
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

logger = get_logger(__name__)

#: Prefix fastText uses to mark label tokens in its training format.
LABEL_PREFIX = "__label__"

__all__ = [
    "LABEL_PREFIX",
    "FastTextClassifier",
    "decode_fasttext_label",
    "encode_fasttext_label",
    "format_training_line",
    "probability_row",
    "sanitise_fasttext_text",
]


def sanitise_fasttext_text(text: Any) -> str:
    """Collapse all whitespace runs so a document fits fastText's line format.

    Args:
        text: Raw document (coerced to ``str``).

    Returns:
        A single-line string with normalised single spaces.
    """
    return " ".join(str(text).split())


def encode_fasttext_label(label: Any) -> str:
    """Encode a label as a fastText ``__label__`` token.

    fastText labels must not contain whitespace, so the label string is
    percent-encoded (losslessly, including ``%`` itself and non-ASCII).

    Args:
        label: Raw label (coerced to ``str``).

    Returns:
        The encoded label token.
    """
    return LABEL_PREFIX + quote(str(label), safe="")


def decode_fasttext_label(encoded: str) -> str:
    """Invert :func:`encode_fasttext_label`.

    Args:
        encoded: A ``__label__``-prefixed token as returned by fastText.

    Returns:
        The original label string.

    Raises:
        DataError: if ``encoded`` does not carry the fastText label prefix.
    """
    if not encoded.startswith(LABEL_PREFIX):
        raise DataError(f"not a fastText label token: {encoded!r}")
    return unquote(encoded[len(LABEL_PREFIX) :])


def format_training_line(text: Any, label: Any) -> str:
    """Format one supervised training example as a fastText input line.

    Args:
        text: Raw document.
        label: Raw label.

    Returns:
        ``"__label__<encoded-label> <single-line text>"``.
    """
    return f"{encode_fasttext_label(label)} {sanitise_fasttext_text(text)}"


def probability_row(
    labels: Sequence[str],
    probabilities: Sequence[float],
    class_to_index: dict[str, int],
    n_classes: int,
) -> np.ndarray:
    """Map fastText's ``(labels, probabilities)`` output onto a fixed class order.

    fastText probabilities can drift marginally outside ``[0, 1]``, so values
    are clipped and the row is renormalised; when fastText returns nothing
    usable (e.g. for an empty document) a uniform distribution is returned so
    rows always sum to one.

    Args:
        labels: Encoded ``__label__`` tokens from ``model.predict``.
        probabilities: Parallel probabilities from ``model.predict``.
        class_to_index: Mapping from decoded label to column index.
        n_classes: Number of columns in the output row.

    Returns:
        A ``float64`` probability row of length ``n_classes`` summing to 1.
    """
    row = np.zeros(n_classes, dtype=np.float64)
    for encoded, probability in zip(labels, probabilities, strict=True):
        index = class_to_index.get(decode_fasttext_label(encoded))
        if index is not None:
            row[index] = min(max(float(probability), 0.0), 1.0)
    total = row.sum()
    if total <= 0.0:
        return np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    return row / total


class FastTextClassifier(ClassifierMixin, BaseEstimator):
    """Supervised fastText text classifier with a scikit-learn interface.

    Follows scikit-learn conventions: ``fit(texts, y)``, ``predict``,
    ``predict_proba``, and a ``classes_`` attribute after fitting. Subword
    n-grams are enabled by default (``minn=2, maxn=5``) because Polish dialect
    markers are often orthographic (mazurzenie respellings, affixes) rather
    than whole-word.

    Attributes:
        classes_: Sorted array of class labels (after ``fit``).
        model_: The trained fastText model (after ``fit``).
    """

    def __init__(
        self,
        *,
        dim: int = 100,
        epoch: int = 25,
        lr: float = 0.5,
        word_ngrams: int = 2,
        min_count: int = 1,
        minn: int = 2,
        maxn: int = 5,
        loss: str = "softmax",
        thread: int = 1,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        """Configure the wrapper; fastText itself is imported only in ``fit``.

        Args:
            dim: Word/subword vector dimensionality.
            epoch: Training epochs (fastText text-classification convention).
            lr: fastText learning rate.
            word_ngrams: Maximum word n-gram length (``wordNgrams``).
            min_count: Minimum token count to keep in the vocabulary.
            minn: Minimum character n-gram length (0 disables subwords).
            maxn: Maximum character n-gram length.
            loss: fastText loss (``"softmax"``, ``"hs"``, or ``"ova"``).
            thread: Trainer threads; the default of 1 keeps training
                deterministic (fastText is non-deterministic when multi-threaded).
            seed: Seed forwarded to fastText when the installed binding
                supports it; otherwise determinism relies on ``thread=1``.
            verbose: fastText's native verbosity (0 silences its C++ output).

        Raises:
            ConfigurationError: if a hyperparameter is out of range.
        """
        if dim < 1:
            raise ConfigurationError(f"dim must be >= 1, got {dim}")
        if epoch < 1:
            raise ConfigurationError(f"epoch must be >= 1, got {epoch}")
        if lr <= 0:
            raise ConfigurationError(f"lr must be > 0, got {lr}")
        if word_ngrams < 1:
            raise ConfigurationError(f"word_ngrams must be >= 1, got {word_ngrams}")
        if min_count < 1:
            raise ConfigurationError(f"min_count must be >= 1, got {min_count}")
        if minn < 0 or maxn < 0 or (minn > maxn and maxn != 0):
            raise ConfigurationError(f"invalid subword range: minn={minn}, maxn={maxn}")
        if thread < 1:
            raise ConfigurationError(f"thread must be >= 1, got {thread}")
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.word_ngrams = word_ngrams
        self.min_count = min_count
        self.minn = minn
        self.maxn = maxn
        self.loss = loss
        self.thread = thread
        self.seed = seed
        self.verbose = verbose

    def fit(self, X: Sequence[str], y: Sequence[Any]) -> FastTextClassifier:
        """Train a supervised fastText model on raw texts.

        Args:
            X: Sequence of raw documents.
            y: Parallel sequence of labels (any hashables; coerced to str).

        Returns:
            ``self``, fitted.

        Raises:
            MissingDependencyError: if fasttext is not installed.
            DataError: if inputs are empty, mismatched, or single-class.
        """
        fasttext = optional_import(
            "fasttext", extra="fasttext", purpose="fastText supervised classification"
        )
        texts = [str(text) for text in X]
        if not texts:
            raise DataError("cannot fit on an empty dataset")
        if len(texts) != len(y):
            raise DataError(f"X and y length mismatch: {len(texts)} != {len(y)}")
        labels = [str(value) for value in y]
        classes = np.unique(np.asarray(labels, dtype=object))
        if len(classes) < 2:
            raise DataError(f"need at least 2 classes to fit, got {len(classes)}")

        train_kwargs: dict[str, Any] = {
            "dim": self.dim,
            "epoch": self.epoch,
            "lr": self.lr,
            "wordNgrams": self.word_ngrams,
            "minCount": self.min_count,
            "minn": self.minn,
            "maxn": self.maxn,
            "loss": self.loss,
            "thread": self.thread,
            "verbose": self.verbose,
        }
        logger.info("training fastText on %d texts / %d classes", len(texts), len(classes))
        with tempfile.TemporaryDirectory(prefix="tulip-fasttext-") as tmp:
            train_path = Path(tmp) / "train.txt"
            with train_path.open("w", encoding="utf-8", newline="\n") as handle:
                for text, label in zip(texts, labels, strict=True):
                    handle.write(format_training_line(text, label))
                    handle.write("\n")
            try:
                model = fasttext.train_supervised(
                    input=str(train_path), seed=self.seed, **train_kwargs
                )
            except (TypeError, ValueError):
                # Older fasttext bindings reject the seed argument; with
                # thread=1 training remains deterministic regardless.
                logger.debug("fasttext.train_supervised rejected seed=%d", self.seed)
                model = fasttext.train_supervised(input=str(train_path), **train_kwargs)
        self.classes_ = classes
        self.model_ = model
        self._class_to_index_ = {str(label): index for index, label in enumerate(classes)}
        return self

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Return the full class distribution for each text.

        Args:
            X: Sequence of raw documents.

        Returns:
            Array of shape ``(len(X), n_classes)``; columns follow ``classes_``.

        Raises:
            TulipError: if the model has not been fitted.
        """
        require_fitted(self, "model_")
        texts = [sanitise_fasttext_text(text) for text in X]
        if not texts:
            return np.zeros((0, len(self.classes_)), dtype=np.float64)
        rows: list[np.ndarray] = []
        for text in texts:
            labels, probabilities = self.model_.predict(text, k=-1, threshold=0.0)
            rows.append(
                probability_row(labels, probabilities, self._class_to_index_, len(self.classes_))
            )
        return np.vstack(rows)

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Return the most probable class label for each text."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


MODELS.add("fasttext", FastTextClassifier)
