"""Shared estimator machinery for the neural and fastText model wrappers.

The helpers the transformer-text, speech, and fastText wrappers share are split
by concern across four modules and re-exported here, so existing imports of
``tulip.models._common`` keep working:

* :mod:`tulip.models._encoding`: label encoding, fit-input validation, class weights.
* :mod:`tulip.models._estimator`: fitted-state checks, validation, the argmax mixin.
* :mod:`tulip.models._factory`: seed reconciliation and checkpoint registry factories.
* :mod:`tulip.models._torch_loops`: the shared torch training and inference loops.

The module stays import-cheap: torch is never imported, and the torch loops
receive the already-imported module from their callers.
"""

from __future__ import annotations

from tulip.models._encoding import (
    balanced_class_weights,
    encode_labels,
    label_id_maps,
    resolve_class_weights,
    validate_fit_inputs,
)
from tulip.models._estimator import (
    ArgmaxPredictMixin,
    require_fitted,
    validate_class_weight,
    validate_common_training_params,
)
from tulip.models._factory import (
    checkpoint_factory,
    reconcile_param_alias,
    reconcile_seed_param,
)
from tulip.models._torch_loops import (
    batched_softmax_probabilities,
    empty_proba,
    linear_warmup_factor,
    optimizer_param_groups,
    resolve_device,
    train_classifier_from_estimator,
    train_torch_classifier,
)

__all__ = [
    "ArgmaxPredictMixin",
    "balanced_class_weights",
    "batched_softmax_probabilities",
    "checkpoint_factory",
    "empty_proba",
    "encode_labels",
    "label_id_maps",
    "linear_warmup_factor",
    "optimizer_param_groups",
    "reconcile_param_alias",
    "reconcile_seed_param",
    "require_fitted",
    "resolve_class_weights",
    "resolve_device",
    "train_classifier_from_estimator",
    "train_torch_classifier",
    "validate_class_weight",
    "validate_common_training_params",
    "validate_fit_inputs",
]
