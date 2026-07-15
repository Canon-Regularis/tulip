"""Registry of text perturbations, built from the generic core registry.

Perturbations self-register under canonical names, so a new stressor is a new
module plus a decorator, never a change here. The registry is a
:class:`tulip.core.registry.Registry` instance, the same generic used by the
model and feature registries; building it here keeps the frozen
``features/registries.py`` untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.core.registry import Registry

if TYPE_CHECKING:
    from tulip.robustness.perturbations import Perturbation

__all__ = ["PERTURBATIONS"]

#: Canonical name -> perturbation class. ``PERTURBATIONS.create(name, **params)``
#: returns a ready perturbation instance.
PERTURBATIONS: Registry[type[Perturbation]] = Registry("perturbation")


def create_perturbation(name: str, **params: Any) -> Perturbation:
    """Instantiate the perturbation registered under ``name``."""
    return PERTURBATIONS.create(name, **params)
