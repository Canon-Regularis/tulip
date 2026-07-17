"""Seed reconciliation and checkpoint-bound registry factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable


def reconcile_param_alias(params: dict[str, Any], alias: str, native: str) -> None:
    """Fold a param spelled ``alias`` onto its ``native`` name, or raise on conflict.

    When ``params`` carries ``alias``, pop it and set ``native`` from it, unless
    ``native`` is already present with a different value. Mutates ``params``. This
    is the one alias-reconciliation rule the neural and fastText factories share,
    so a config can swap ``model.name`` between trainable models without renaming
    its params.

    Raises:
        ConfigurationError: if ``alias`` and ``native`` are both present and
            disagree.
    """
    if alias not in params:
        return
    value = params.pop(alias)
    if native in params and params[native] != value:
        raise ConfigurationError(
            f"conflicting values: {alias}={value!r} vs {native}={params[native]!r}"
        )
    params.setdefault(native, value)


def reconcile_seed_param(params: dict[str, Any]) -> None:
    """Map scikit-learn's ``random_state`` spelling onto the wrappers' ``seed``.

    The classical factories accept both spellings (see
    ``tulip.models.classical``); this keeps the neural/fastText factories
    interchangeable with them in experiment configs. Mutates ``params``.

    Raises:
        ConfigurationError: if ``random_state`` and ``seed`` disagree.
    """
    reconcile_param_alias(params, "random_state", "seed")


def pop_seed(params: dict[str, Any], *, default: int | None) -> int | None:
    """Reconcile the two seed spellings and pop the resolved seed out of ``params``.

    The single seed-extraction path for the factories that pass a concrete seed
    to their estimator (the classical and ensemble factories): it reuses
    :func:`reconcile_seed_param` for the ``random_state``/``seed`` reconciliation
    (and its conflict check), then removes and returns the seed.

    Args:
        params: Factory keyword arguments; mutated in place (both spellings gone).
        default: Seed returned when neither spelling is present.

    Returns:
        The resolved seed (may be ``None`` to request unseeded behaviour).

    Raises:
        ConfigurationError: if ``random_state`` and ``seed`` disagree.
    """
    reconcile_seed_param(params)
    return params.pop("seed", default)


def checkpoint_factory(cls: type, checkpoint: str) -> Callable[..., Any]:
    """Build a registry factory that pre-binds a default checkpoint.

    The bound checkpoint (and every other constructor parameter) remains
    overridable through the factory's keyword arguments, so experiment configs
    can swap checkpoints without registering new names. ``random_state`` is
    accepted as an alias for ``seed`` (scikit-learn spelling).

    Args:
        cls: The wrapper class to instantiate.
        checkpoint: Default Hugging Face checkpoint or model source.

    Returns:
        A keyword-only factory suitable for ``Registry.add``.
    """

    def factory(**params: Any) -> Any:
        params.setdefault("checkpoint", checkpoint)
        reconcile_seed_param(params)
        return cls(**params)

    safe = checkpoint.replace("/", "_").replace("-", "_").replace(".", "_")
    factory.__name__ = f"make_{cls.__name__.lower()}_{safe}"
    factory.__qualname__ = factory.__name__
    factory.__doc__ = f"Create a :class:`{cls.__name__}` pre-bound to checkpoint {checkpoint!r}."
    return factory
