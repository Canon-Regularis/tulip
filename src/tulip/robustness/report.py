"""Configs and reports for robustness under linguistic perturbation.

A robustness sweep trains one model, then scores it as the test inputs are
perturbed along an intensity axis. The result is a grid of macro-F1 by
perturbation and level, plus a clean baseline. These models own that grid.

Every config is a module-owned pydantic model, so nothing here is bolted onto
the frozen ``ExperimentConfig``. The report serialises through
:func:`tulip._serialize.write_sorted_json` with a fixed float precision, so a
committed ``robustness-<name>.json`` regenerates byte for byte. The markdown
helpers are imported lazily inside :meth:`RobustnessReport.to_markdown`, so
importing this module pulls no sklearn or torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tulip._serialize import write_sorted_json

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "BREAKDOWN_FRACTION",
    "ROBUSTNESS_FLOAT_DIGITS",
    "AugmentSpec",
    "PerturbationConfig",
    "RobustnessCell",
    "RobustnessCurve",
    "RobustnessReport",
]

#: Digits kept in the JSON artifact, matching the leaderboard provenance so a
#: committed robustness report stays byte-identical across re-runs.
ROBUSTNESS_FLOAT_DIGITS = 6

#: A curve breaks down at the first level whose macro-F1 falls below this
#: fraction of the clean score.
BREAKDOWN_FRACTION = 0.8


class PerturbationConfig(BaseModel):
    """One perturbation to sweep: a registered name, its levels, and a seed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    seed: int = 0
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("levels")
    @classmethod
    def _levels_in_unit_range(cls, levels: tuple[float, ...]) -> tuple[float, ...]:
        if not levels:
            raise ValueError("levels must be non-empty")
        if any(not 0.0 <= level <= 1.0 for level in levels):
            raise ValueError("every level must lie within [0, 1]")
        return levels


class RobustnessCell(BaseModel):
    """One model score at one perturbation and one intensity level."""

    model_config = ConfigDict(frozen=True)

    perturbation: str
    level: float = Field(ge=0.0, le=1.0)
    n: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)


class RobustnessCurve(BaseModel):
    """One perturbation swept across its levels, against the clean score."""

    model_config = ConfigDict(frozen=True)

    perturbation: str
    clean_f1: float = Field(ge=0.0, le=1.0)
    cells: tuple[RobustnessCell, ...]

    @property
    def levels(self) -> tuple[float, ...]:
        """The swept levels in cell order."""
        return tuple(cell.level for cell in self.cells)

    @property
    def degradation_slope(self) -> float:
        """Endpoint slope of macro-F1 across the swept levels (0.0 if flat)."""
        if len(self.cells) < 2:
            return 0.0
        first, last = self.cells[0], self.cells[-1]
        span = last.level - first.level
        if span == 0.0:
            return 0.0
        return (last.f1_macro - first.f1_macro) / span

    @property
    def breakdown_level(self) -> float | None:
        """First level whose macro-F1 drops below :data:`BREAKDOWN_FRACTION` of clean."""
        threshold = BREAKDOWN_FRACTION * self.clean_f1
        for cell in self.cells:
            if cell.f1_macro < threshold:
                return cell.level
        return None


class RobustnessReport(BaseModel):
    """A model's macro-F1 grid over perturbations and levels, plus the baseline."""

    model_config = ConfigDict(frozen=True)

    model: str
    target: str
    baseline: RobustnessCell
    curves: tuple[RobustnessCurve, ...]

    def to_markdown(self) -> str:
        """Render a perturbation-by-level macro-F1 grid, modelled on TransferMatrix."""
        from tulip.evaluation._format import format_metric, markdown_table

        levels = sorted({level for curve in self.curves for level in curve.levels})
        headers = ("perturbation \\ level", *(format_metric(level, digits=2) for level in levels))
        rows = [
            (
                curve.perturbation,
                *(
                    format_metric({cell.level: cell.f1_macro for cell in curve.cells}.get(level))
                    for level in levels
                ),
            )
            for curve in self.curves
        ]
        title = f"# Robustness - {self.model} ({self.target})"
        note = (
            f"Clean macro-F1: {format_metric(self.baseline.f1_macro)} "
            f"over {self.baseline.n} samples"
        )
        return f"{title}\n\n{note}\n\n{markdown_table(headers, rows)}"

    def save(self, path: Path) -> None:
        """Write the report as deterministic JSON, derived slopes included."""
        payload = {
            "model": self.model,
            "target": self.target,
            "baseline": self.baseline.model_dump(mode="json"),
            "curves": [
                {
                    "perturbation": curve.perturbation,
                    "clean_f1": curve.clean_f1,
                    "degradation_slope": curve.degradation_slope,
                    "breakdown_level": curve.breakdown_level,
                    "cells": [cell.model_dump(mode="json") for cell in curve.cells],
                }
                for curve in self.curves
            ],
        }
        write_sorted_json(path, _round_floats(payload, ROBUSTNESS_FLOAT_DIGITS))


class AugmentSpec(BaseModel):
    """How to augment a training set: which perturbations, how many copies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    perturbations: tuple[PerturbationConfig, ...]
    multiplier: int = Field(default=1, ge=1)
    seed: int = 0


def _round_floats(value: Any, digits: int) -> Any:
    """Recursively round floats so the serialised artifact is byte-stable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item, digits) for item in value]
    return value
