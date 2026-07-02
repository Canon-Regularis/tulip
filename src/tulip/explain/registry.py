"""Registry of explanation methods.

Explainers are registered under canonical names (``top_tfidf``, ``lime``,
``shap``, ``attention``, ``nearest_examples``) and looked up by experiment
configs and the CLI purely by name. Every registered component is a class
whose instances satisfy :class:`tulip.core.interfaces.Explainer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.core.registry import Registry

if TYPE_CHECKING:
    from tulip.core.interfaces import Explainer

#: Explanation methods (token/feature attributions, neighbours, attention).
EXPLAINERS: Registry[Any] = Registry("explainer")


def get_explainer(name: str, **kwargs: Any) -> Explainer:
    """Instantiate the explainer registered under ``name``.

    Args:
        name: Canonical registry name (e.g. ``"top_tfidf"``) or alias.
        **kwargs: Constructor parameters forwarded to the explainer class.

    Returns:
        A ready-to-use explainer instance.

    Raises:
        UnknownComponentError: if no explainer is registered under ``name``.
    """
    return EXPLAINERS.create(name, **kwargs)
