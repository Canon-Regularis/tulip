"""Homogeneous ensembles over the registered feature-based models.

Registers two factories in :data:`tulip.models.MODELS`:

* ``voting`` — a soft- (or hard-) voting ensemble that averages the base models'
  probabilities.
* ``stacking`` — out-of-fold stacking, where the base models' cross-validated
  probabilities train a meta-learner.

Both combine several registered models that share one feature matrix into a
single estimator satisfying the :class:`~tulip.core.interfaces.Classifier`
contract, so an ensemble is usable anywhere a model name is (a config's ``model``
field, ``tulip benchmark -m voting``, ``DialectClassifier(model="voting")``).

The ensembling logic is not reimplemented. These factories reuse scikit-learn's
:class:`~sklearn.ensemble.VotingClassifier` and
:class:`~sklearn.ensemble.StackingClassifier`, and build the base estimators
through :meth:`MODELS.create`. The only new code is the glue that turns a list of
registry names into unique ``(alias, estimator)`` pairs and threads the seed
through. Seed handling reuses ``_pop_seed`` from
:mod:`tulip.models.classical`, so the two spellings (``random_state``/``seed``)
behave identically to every other baseline.

The bases must consume the same feature matrix as any classical model. Base-model
constraints carry over: soft voting and stacking need bases with
``predict_proba``, and ``naive_bayes`` still needs non-negative features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.core.exceptions import ConfigurationError
from tulip.models.classical import DEFAULT_SEED, _pop_seed
from tulip.models.registry import MODELS
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

__all__ = ["make_stacking", "make_voting"]

#: Meta-learner used by ``stacking`` when none is named.
_DEFAULT_FINAL = "logistic_regression"


@MODELS.register("voting")
def make_voting(
    *,
    estimators: Sequence[Any],
    weights: Sequence[float] | None = None,
    voting: str = "soft",
    **params: Any,
) -> Any:
    """Voting ensemble over registered models sharing a feature matrix.

    Args:
        estimators: Base models. Each entry is a registry name, or a mapping
            ``{"name": ..., "alias": ..., "params": {...}}`` for per-base
            overrides.
        weights: Optional per-estimator weights (soft voting); length must match
            ``estimators``.
        voting: ``"soft"`` (average probabilities, the default) or ``"hard"``
            (majority label). Soft voting needs bases with ``predict_proba``.
        **params: Accepts ``random_state``/``seed``, threaded into every base.

    Returns:
        An unfitted :class:`sklearn.ensemble.VotingClassifier`.

    Raises:
        ConfigurationError: if ``estimators`` is empty, ``voting`` is unknown,
            ``weights`` length mismatches, or unexpected params are given.
    """
    from sklearn.ensemble import VotingClassifier

    seed = _pop_seed(params, default=None)
    _reject_extra(params, "voting")
    if voting not in ("soft", "hard"):
        raise ConfigurationError(f"voting must be 'soft' or 'hard', got {voting!r}")
    bases = _build_bases(estimators, seed)
    weight_list = list(weights) if weights is not None else None
    if weight_list is not None and len(weight_list) != len(bases):
        raise ConfigurationError(
            f"weights has {len(weight_list)} entries but there are {len(bases)} estimators"
        )
    return VotingClassifier(estimators=bases, voting=voting, weights=weight_list)


@MODELS.register("stacking")
def make_stacking(
    *,
    estimators: Sequence[Any],
    final_estimator: str = _DEFAULT_FINAL,
    cv: int = 5,
    **params: Any,
) -> Any:
    """Out-of-fold stacking ensemble over registered models sharing features.

    The base models are cross-validated to produce out-of-fold probabilities,
    which train the ``final_estimator`` meta-learner. This avoids the leakage of
    training the meta-learner on in-sample base predictions.

    Args:
        estimators: Base models, in the same form as :func:`make_voting`.
        final_estimator: Registry name of the meta-learner (default
            ``logistic_regression``).
        cv: Number of stratified folds for the out-of-fold base predictions.
        **params: Accepts ``random_state``/``seed``, threaded into the bases, the
            meta-learner, and the fold shuffling.

    Returns:
        An unfitted :class:`sklearn.ensemble.StackingClassifier`.

    Raises:
        ConfigurationError: if ``estimators`` is empty, ``cv < 2``, or unexpected
            params are given.
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.model_selection import StratifiedKFold

    seed = _pop_seed(params, default=DEFAULT_SEED)
    _reject_extra(params, "stacking")
    if cv < 2:
        raise ConfigurationError(f"stacking cv must be >= 2, got {cv}")
    bases = _build_bases(estimators, seed)
    final = MODELS.create(final_estimator, seed=seed)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    return StackingClassifier(
        estimators=bases,
        final_estimator=final,
        cv=splitter,
        stack_method="predict_proba",
    )


def _build_bases(estimators: Sequence[Any], seed: int | None) -> list[tuple[str, Any]]:
    """Turn base-model references into unique ``(alias, estimator)`` pairs."""
    if not estimators:
        raise ConfigurationError("an ensemble needs at least one base estimator")
    pairs: list[tuple[str, Any]] = []
    used: set[str] = set()
    for entry in estimators:
        name, alias, base_params = _parse_entry(entry)
        # Thread the ensemble seed into every base unless the base overrides it,
        # so a stochastic base is reproducible and consistent across the ensemble.
        merged = {"seed": seed, **base_params} if seed is not None else dict(base_params)
        pairs.append((_unique(alias, used), MODELS.create(name, **merged)))
    return pairs


def _parse_entry(entry: Any) -> tuple[str, str, dict[str, Any]]:
    """Normalise one base entry into ``(name, alias, params)``."""
    if isinstance(entry, str):
        return entry, entry, {}
    if isinstance(entry, dict):
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigurationError(f"ensemble estimator mapping needs a 'name': {entry!r}")
        alias = str(entry.get("alias", name))
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ConfigurationError(f"ensemble estimator 'params' must be a mapping: {entry!r}")
        return name, alias, dict(params)
    raise ConfigurationError(
        f"ensemble estimator must be a name or a mapping, got {type(entry).__name__}"
    )


def _unique(alias: str, used: set[str]) -> str:
    """Return ``alias`` made unique against ``used`` (append ``_2``, ``_3``, …)."""
    candidate, index = alias, 2
    while candidate in used:
        candidate = f"{alias}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _reject_extra(params: dict[str, Any], name: str) -> None:
    """Raise if any unexpected keyword params remain after the known ones."""
    if params:
        raise ConfigurationError(f"unexpected {name} params: {sorted(params)}")
