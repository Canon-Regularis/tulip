"""Compare text-only, audio-only, and fused dialect classification.

The multimodal question a leaderboard does not answer: does combining the text and
audio experts actually beat the better of the two alone? This trains nothing new.
It takes a fitted text base and a fitted audio base, evaluates each on its own and
their fusion on one identical multimodal test set (the samples that carry both a
text and an audio input plus a gold label), and reports the three side by side
with the fusion uplift and a paired McNemar test against each single modality, so
"fusion helps" is a measured, significance-backed claim rather than an assumption.

It composes :class:`~tulip.pipeline.fusion.classifier.MultimodalClassifier` with the
evaluation metrics and significance test, so it lives in ``pipeline`` (which
depends on ``evaluation``), not the other way round.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import format_metric, markdown_table, save_report
from tulip.core.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tulip.core.types import Sample
    from tulip.pipeline.fusion.strategies import FusionStrategy
    from tulip.pipeline.protocols import ProbabilisticClassifier

__all__ = ["ModalityComparison", "ModalityScore", "compare_modalities"]

#: Stored floats are rounded to this many digits so a saved report is byte-stable.
FUSION_COMPARISON_FLOAT_DIGITS = 6


class ModalityScore(BaseModel):
    """One condition's accuracy and F1 on the shared multimodal test set."""

    model_config = ConfigDict(frozen=True)

    modality: str
    n_correct: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)
    f1_weighted: float = Field(ge=0.0, le=1.0)


class ModalityComparison(BaseModel):
    """Text-only vs audio-only vs fused, with the fusion uplift and its significance."""

    model_config = ConfigDict(frozen=True)

    target: str
    strategy: str
    n_samples: int = Field(ge=1)
    text: ModalityScore
    audio: ModalityScore
    fused: ModalityScore
    uplift_f1_macro: float
    fused_beats_best_single: bool
    mcnemar_p_vs_text: float = Field(ge=0.0, le=1.0)
    mcnemar_p_vs_audio: float = Field(ge=0.0, le=1.0)

    def to_markdown(self) -> str:
        """Render the comparison table with the fusion verdict and significance."""
        title = f"# Modality comparison ({self.target}, fusion={self.strategy})"
        verdict = (
            "Fusion beats" if self.fused_beats_best_single else "Fusion does not beat"
        ) + f" the best single modality by {format_metric(self.uplift_f1_macro)} macro F1."
        significance = (
            f"Paired McNemar (fused vs single): vs text p={format_metric(self.mcnemar_p_vs_text)}, "
            f"vs audio p={format_metric(self.mcnemar_p_vs_audio)}."
        )
        rows = [
            (
                score.modality,
                format_metric(score.accuracy),
                format_metric(score.f1_macro),
                format_metric(score.f1_weighted),
            )
            for score in (self.text, self.audio, self.fused)
        ]
        headers = ("Modality", "Accuracy", "F1 (macro)", "F1 (weighted)")
        return (
            f"{title}\n\n{self.n_samples} multimodal test samples. {verdict} {significance}"
            f"\n\n{markdown_table(headers, rows)}"
        )

    def save(self, path: Path | str) -> None:
        """Write the report as deterministic JSON (sorted keys, rounded floats)."""
        save_report(self, path, digits=FUSION_COMPARISON_FLOAT_DIGITS)


def compare_modalities(
    text: ProbabilisticClassifier,
    audio: ProbabilisticClassifier,
    samples: Sequence[Sample],
    *,
    strategy: FusionStrategy | None = None,
) -> ModalityComparison:
    """Evaluate text-only, audio-only, and their fusion on one multimodal split.

    Args:
        text: A fitted text-modality classifier (``task == TEXT``).
        audio: A fitted audio-modality classifier (``task == AUDIO``).
        samples: Evaluation samples; only those carrying both a text and an audio
            input and a label at the shared target level are scored.
        strategy: The fusion strategy; defaults to an equal weighted average.

    Returns:
        A :class:`ModalityComparison`.

    Raises:
        ConfigurationError: if the two bases disagree on target or carry the wrong
            modality task (raised by :class:`MultimodalClassifier`).
        DataError: if no sample carries text, audio, and a target-level label.
    """
    from tulip.evaluation.significance import mcnemar_exact
    from tulip.pipeline.fusion.classifier import MultimodalClassifier

    fused_classifier = MultimodalClassifier(text=text, audio=audio, strategy=strategy)
    target = fused_classifier.target

    usable = [
        sample
        for sample in samples
        if sample.text is not None
        and sample.audio_path is not None
        and sample.labels.at_level(target) is not None
    ]
    if not usable:
        raise DataError(
            f"no samples carry text, audio, and a label at target {target.value!r}; "
            "a modality comparison needs a multimodal, labelled test set"
        )
    gold = [str(sample.labels.at_level(target)) for sample in usable]

    text_pred = _argmax_labels(text, [str(sample.text) for sample in usable])
    audio_pred = _argmax_labels(audio, [str(sample.audio_path) for sample in usable])
    fused_proba = fused_classifier.predict_proba_samples(usable)
    fused_pred = [fused_classifier.classes_[int(index)] for index in fused_proba.argmax(axis=1)]

    text_score = _score("text", gold, text_pred)
    audio_score = _score("audio", gold, audio_pred)
    fused_score = _score("text+audio", gold, fused_pred)

    fused_correct = _correct(fused_pred, gold)
    _, _, p_text = mcnemar_exact(fused_correct, _correct(text_pred, gold))
    _, _, p_audio = mcnemar_exact(fused_correct, _correct(audio_pred, gold))

    best_single = max(text_score.f1_macro, audio_score.f1_macro)
    return ModalityComparison(
        target=target.value,
        strategy=fused_classifier.strategy.kind,
        n_samples=len(usable),
        text=text_score,
        audio=audio_score,
        fused=fused_score,
        uplift_f1_macro=fused_score.f1_macro - best_single,
        fused_beats_best_single=fused_score.f1_macro > best_single,
        mcnemar_p_vs_text=p_text,
        mcnemar_p_vs_audio=p_audio,
    )


def _argmax_labels(classifier: ProbabilisticClassifier, raws: Sequence[str]) -> list[str]:
    """Argmax predicted labels from a classifier's probability matrix."""
    proba = np.asarray(classifier.predict_proba(raws), dtype=np.float64)
    return [classifier.classes_[int(index)] for index in proba.argmax(axis=1)]


def _correct(predictions: Sequence[str], gold: Sequence[str]) -> list[bool]:
    """Per-sample correctness, aligned to ``gold``."""
    return [pred == truth for pred, truth in zip(predictions, gold, strict=True)]


def _score(modality: str, gold: Sequence[str], predictions: Sequence[str]) -> ModalityScore:
    """Metrics for one condition on the shared test set."""
    from tulip.evaluation.metrics import compute_metrics

    report = compute_metrics(gold, predictions)
    n_correct = sum(1 for pred, truth in zip(predictions, gold, strict=True) if pred == truth)
    return ModalityScore(
        modality=modality,
        n_correct=n_correct,
        accuracy=report.accuracy,
        f1_macro=report.f1_macro,
        f1_weighted=report.f1_weighted,
    )
