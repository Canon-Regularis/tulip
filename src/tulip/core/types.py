"""Canonical data types flowing through every tulip pipeline.

These pydantic models define the contract between subsystems: dataset loaders
produce :class:`Sample` objects, classifiers produce :class:`Prediction`
objects, and explainers produce :class:`Explanation` objects.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tulip.labels.taxonomy import DialectFamily, LabelLevel, family_for


class TaskType(str, enum.Enum):
    """Input modality of a classification task."""

    TEXT = "text"
    AUDIO = "audio"


class DialectLabels(BaseModel):
    """Hierarchical gold labels attached to a sample.

    Any level may be missing; loaders fill in what their corpus provides.
    ``family`` is derived from ``dialect`` automatically when absent.
    """

    model_config = ConfigDict(frozen=True)

    family: str | None = None
    dialect: str | None = None
    region: str | None = None
    village: str | None = None
    voivodeship: str | None = None

    @model_validator(mode="after")
    def _derive_family(self) -> DialectLabels:
        if self.family is None and self.dialect is not None:
            derived = family_for(self.dialect)
            if derived is not None:
                object.__setattr__(self, "family", derived.value)
        return self

    def at_level(self, level: LabelLevel) -> str | None:
        """Return the label at the requested granularity, or ``None`` if absent."""
        return getattr(self, level.value)

    def is_standard(self) -> bool:
        """Whether this sample is labelled as standard (non-dialectal) Polish."""
        return self.family == DialectFamily.STANDARD.value


class Sample(BaseModel):
    """A single labelled unit of text and/or audio.

    At least one of ``text`` or ``audio_path`` must be present. ``speaker_id``
    is required for leakage-free (speaker-disjoint) train/test splitting;
    loaders should synthesise a stable surrogate ID when the corpus does not
    identify speakers.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str | None = None
    audio_path: Path | None = None
    speaker_id: str | None = None
    labels: DialectLabels = Field(default_factory=DialectLabels)
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_content(self) -> Sample:
        if self.text is None and self.audio_path is None:
            raise ValueError(f"sample {self.id!r} has neither text nor audio_path")
        return self


class ClassProbability(BaseModel):
    """One class with its predicted probability."""

    model_config = ConfigDict(frozen=True)

    label: str
    probability: float = Field(ge=0.0, le=1.0)


class Prediction(BaseModel):
    """The result of classifying one sample.

    ``probabilities`` is always sorted by descending probability. ``label`` is
    ``None`` when the classifier abstained (uncertainty above the configured
    threshold), never silently absent.
    """

    model_config = ConfigDict(frozen=True)

    label: str | None
    level: LabelLevel = LabelLevel.DIALECT
    probabilities: tuple[ClassProbability, ...] = ()
    abstained: bool = False

    @model_validator(mode="after")
    def _sort_probabilities(self) -> Prediction:
        ordered = tuple(sorted(self.probabilities, key=lambda cp: cp.probability, reverse=True))
        object.__setattr__(self, "probabilities", ordered)
        return self

    @property
    def confidence(self) -> float:
        """Probability of the top class (0.0 when no probabilities are available)."""
        return self.probabilities[0].probability if self.probabilities else 0.0

    def top_k(self, k: int = 3) -> tuple[ClassProbability, ...]:
        """Return the ``k`` most probable classes."""
        return self.probabilities[:k]

    def as_dict(self) -> dict[str, float]:
        """Return probabilities as a plain ``{label: probability}`` mapping."""
        return {cp.label: cp.probability for cp in self.probabilities}


class TokenAttribution(BaseModel):
    """How strongly one token/feature pushed the prediction towards a class."""

    model_config = ConfigDict(frozen=True)

    token: str
    weight: float


class NeighborExample(BaseModel):
    """A similar training example retrieved as supporting evidence."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    label: str | None = None
    text: str | None = None
    similarity: float = 0.0


class Explanation(BaseModel):
    """Why a prediction was made, in a method-agnostic shape.

    ``attributions`` covers token/feature-level evidence (SHAP, LIME, linear
    coefficients, attention); ``neighbors`` covers example-level evidence;
    ``details`` carries method-specific extras (e.g. attention matrices).
    """

    model_config = ConfigDict(frozen=True)

    method: str
    predicted_label: str | None = None
    attributions: tuple[TokenAttribution, ...] = ()
    neighbors: tuple[NeighborExample, ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)

    def top_attributions(self, k: int = 10) -> tuple[TokenAttribution, ...]:
        """Return the ``k`` attributions with the largest absolute weight."""
        ordered = sorted(self.attributions, key=lambda a: abs(a.weight), reverse=True)
        return tuple(ordered[:k])


class DatasetInfo(BaseModel):
    """Static metadata describing a source corpus."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    url: str = ""
    tier: int = Field(ge=1, le=4, default=4)
    tasks: tuple[str, ...] = ()
    contents: tuple[str, ...] = ()
    label_levels: tuple[LabelLevel, ...] = ()
    license: str = "unknown; check the source before redistribution"
