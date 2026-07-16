"""Seed reconciliation and checkpoint-bound registry factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable


def reconcile_seed_param(params: dict[str, Any]) -> None:
    """Map scikit-learn's ``random_state`` spelling onto the wrappers' ``seed``.

    The classical factories accept both spellings (see
    ``tulip.models.classical``); this keeps the neural/fastText factories
    interchangeable with them in experiment configs. Mutates ``params``.

    Raises:
        ConfigurationError: if ``random_state`` and ``seed`` disagree.
    """
    if "random_state" not in params:
        return
    random_state = params.pop("random_state")
    if "seed" in params and params["seed"] != random_state:
        raise ConfigurationError(
            f"conflicting seeds: random_state={random_state!r} vs seed={params['seed']!r}"
        )
    params.setdefault("seed", random_state)


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
