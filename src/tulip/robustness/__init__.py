"""Robustness under linguistic perturbation.

One seeded perturbation engine, grounded in the Polish phonological rules,
scores a model as its inputs move along an intensity axis. The same engine
augments training data. Perturbations self-register in :data:`PERTURBATIONS` on
import.
"""

from __future__ import annotations

from tulip.robustness.registry import PERTURBATIONS
from tulip.robustness.report import (
    AugmentSpec,
    PerturbationConfig,
    RobustnessCell,
    RobustnessCurve,
    RobustnessReport,
)
from tulip.robustness.sweep import perturb_samples, run_robustness


def _register_builtins() -> None:
    from tulip.robustness import perturbations  # noqa: F401


_register_builtins()

__all__ = [
    "PERTURBATIONS",
    "AugmentSpec",
    "PerturbationConfig",
    "RobustnessCell",
    "RobustnessCurve",
    "RobustnessReport",
    "perturb_samples",
    "run_robustness",
]
