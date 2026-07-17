"""Explainability: why a dialect prediction was made.

Importing this package registers all built-in explainers in
:data:`EXPLAINERS` under their canonical names (``top_tfidf``, ``lime``,
``shap``, ``attention``, ``nearest_examples``, ``dialect_evidence``). Explainers
with heavy dependencies (lime, shap, torch) import them lazily inside
``explain``, so registration itself is always cheap.
"""

from __future__ import annotations

from tulip.explain.aggregate import (
    ClassCount,
    FamilyEvidence,
    GlobalEvidenceReport,
    PhenomenonFrequency,
    dataset_evidence,
)
from tulip.explain.contrast import ContrastFeature, ContrastReport, contrast_dialects
from tulip.explain.registry import EXPLAINERS, get_explainer


def _register_builtins() -> None:
    from tulip.explain import (  # noqa: F401
        attention,
        dialect_evidence,
        lime_explainer,
        linear,
        neighbors,
        shap_explainer,
    )


_register_builtins()

__all__ = [
    "EXPLAINERS",
    "ClassCount",
    "ContrastFeature",
    "ContrastReport",
    "FamilyEvidence",
    "GlobalEvidenceReport",
    "PhenomenonFrequency",
    "contrast_dialects",
    "dataset_evidence",
    "get_explainer",
]
