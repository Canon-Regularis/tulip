"""Pydantic schemas for experiment configuration.

Experiments are declared in YAML and validated into these models. Components
(datasets, features, models, explainers) are referenced by their registry
names plus free-form ``params``, which keeps the config schema stable as new
components are added.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tulip.core.types import TaskType
from tulip.labels.taxonomy import LabelLevel


class ComponentConfig(BaseModel):
    """A registry component reference: its canonical name plus constructor params."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class SplitConfig(BaseModel):
    """Speaker-disjoint train/validation/test split proportions."""

    model_config = ConfigDict(extra="forbid")

    train: float = Field(default=0.70, gt=0.0, lt=1.0)
    validation: float = Field(default=0.15, ge=0.0, lt=1.0)
    test: float = Field(default=0.15, gt=0.0, lt=1.0)
    group_by: str = "speaker_id"
    stratify_by: LabelLevel | None = LabelLevel.DIALECT
    seed: int = 42

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> SplitConfig:
        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"split fractions must sum to 1.0, got {total:.4f}")
        return self


class DataConfig(BaseModel):
    """Which corpora to use and how to prepare them."""

    model_config = ConfigDict(extra="forbid")

    datasets: list[ComponentConfig] = Field(min_length=1)
    root: Path = Path("data/raw")
    clean: bool = True
    deduplicate: bool = True
    min_text_chars: int = Field(default=20, ge=0)


class TrainingConfig(BaseModel):
    """Knobs shared across trainable models; model-specific ones go in model params."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0.0)


class ExperimentConfig(BaseModel):
    """A complete, reproducible experiment declaration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 42
    task: TaskType = TaskType.TEXT
    target: LabelLevel = LabelLevel.DIALECT
    data: DataConfig
    features: list[ComponentConfig] = Field(default_factory=list)
    model: ComponentConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    output_dir: Path = Path("artifacts")
    abstain_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
