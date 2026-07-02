"""Explainability: why a dialect prediction was made.

Importing this package registers all built-in explainers in
:data:`EXPLAINERS` under their canonical names (``top_tfidf``, ``lime``,
``shap``, ``attention``, ``nearest_examples``). Explainers with heavy
dependencies (lime, shap, torch) import them lazily inside ``explain``, so
registration itself is always cheap.
"""

from __future__ import annotations

from tulip.explain.registry import EXPLAINERS, get_explainer


def _register_builtins() -> None:
    from tulip.explain import (  # noqa: F401
        attention,
        lime_explainer,
        linear,
        neighbors,
        shap_explainer,
    )


_register_builtins()

__all__ = ["EXPLAINERS", "get_explainer"]
