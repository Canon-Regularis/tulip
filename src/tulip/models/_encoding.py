"""Label encoding, fit-input validation, and class-weight helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tulip.core.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Sequence


def encode_labels(y: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Encode arbitrary labels as integer ids over a sorted class vocabulary.

    Args:
        y: Sequence of labels (coerced to ``str``).

    Returns:
        ``(classes, encoded)`` where ``classes`` is the sorted array of unique
        label strings and ``encoded`` is an ``int64`` array with
        ``classes[encoded[i]] == str(y[i])``.
    """
    labels = np.asarray([str(value) for value in y], dtype=object)
    classes, encoded = np.unique(labels, return_inverse=True)
    return classes, encoded.astype(np.int64)


def validate_fit_inputs(inputs: Sequence[Any], y: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Validate the universal ``fit(X, y)`` preconditions and encode ``y``.

    Every wrapper shares the same three requirements: a non-empty ``X``,
    aligned lengths, and at least two classes.

    Args:
        inputs: The raw model inputs (texts or audio paths), already coerced.
        y: Parallel label sequence.

    Returns:
        ``(classes, encoded)`` as from :func:`encode_labels`.

    Raises:
        DataError: if inputs are empty, mismatched, or single-class.
    """
    if not inputs:
        raise DataError("cannot fit on an empty dataset")
    if len(inputs) != len(y):
        raise DataError(f"X and y length mismatch: {len(inputs)} != {len(y)}")
    classes, encoded = encode_labels(y)
    if len(classes) < 2:
        raise DataError(f"need at least 2 classes to fit, got {len(classes)}")
    return classes, encoded


def balanced_class_weights(encoded: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute sklearn-style ``"balanced"`` class weights from encoded labels.

    Uses ``n_samples / (n_classes * count_per_class)`` so rare dialects are
    not drowned out by majority classes during fine-tuning.

    Args:
        encoded: Integer label ids in ``[0, n_classes)``.
        n_classes: Total number of classes.

    Returns:
        A ``float64`` array of per-class weights of length ``n_classes``.
    """
    counts = np.bincount(np.asarray(encoded, dtype=np.int64), minlength=n_classes)
    counts = np.maximum(counts, 1)  # guard classes absent from this fold
    return len(encoded) / (n_classes * counts.astype(np.float64))


def label_id_maps(classes: Sequence[Any] | np.ndarray) -> tuple[dict[int, str], dict[str, int]]:
    """Build the ``id2label`` / ``label2id`` maps a Hugging Face head expects.

    Both fine-tuning wrappers pass these to ``from_pretrained`` so the fitted
    model carries human-readable labels. The label keys use the same
    ``str(label)`` coercion as :func:`encode_labels`, so the maps stay aligned
    with the ``classes`` array.

    Args:
        classes: The sorted class array from :func:`encode_labels`.

    Returns:
        ``(id2label, label2id)``: mutually inverse dicts over the class ids.
    """
    id2label = {index: str(label) for index, label in enumerate(classes)}
    label2id = {label: index for index, label in id2label.items()}
    return id2label, label2id


def resolve_class_weights(estimator: Any, encoded: np.ndarray, n_classes: int) -> np.ndarray | None:
    """Return balanced loss weights when the estimator asks for them, else ``None``.

    Centralises the ``class_weight == "balanced"`` branch every torch wrapper
    runs before training, so the loss-weighting policy lives in one place.

    Args:
        estimator: The wrapper whose ``class_weight`` selects the policy.
        encoded: Integer label ids in ``[0, n_classes)``.
        n_classes: Total number of classes.

    Returns:
        Per-class weights from :func:`balanced_class_weights`, or ``None`` when
        ``class_weight`` is not ``"balanced"``.
    """
    if estimator.class_weight == "balanced":
        return balanced_class_weights(encoded, n_classes)
    return None
