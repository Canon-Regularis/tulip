"""Tests for the text vs audio vs fused modality comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tulip.core.exceptions import DataError
from tulip.core.types import DialectLabels, Sample, TaskType
from tulip.labels.taxonomy import LabelLevel
from tulip.pipeline.fusion import ModalityComparison, WeightedAverageFusion, compare_modalities

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: Distinctive multi-character labels so an incidental letter in a raw string
#: cannot look like a class name. "west" is never a gold label, so it is the
#: deliberate "no signal" fallback that makes a signalless modality wrong.
_CLASSES = ("north", "south", "west")


class _KeyedStub:
    """A ProbabilisticClassifier that predicts the class name found in the raw.

    A raw containing no class name falls back to ``west`` (never a gold label
    here), so a modality that lacks signal on a sample is simply wrong there. This
    lets a test construct genuinely complementary experts.
    """

    def __init__(self, task: TaskType) -> None:
        self.classes_ = _CLASSES
        self.task = task
        self.target = LabelLevel.DIALECT

    def predict_proba(self, raws: Sequence[Any]) -> np.ndarray:
        rows = []
        for raw in raws:
            text = str(raw)
            weights = np.full(len(self.classes_), 0.05)
            matched = [i for i, label in enumerate(self.classes_) if label in text]
            weights[matched[0] if matched else -1] = 0.9
            rows.append(weights / weights.sum())
        return np.asarray(rows)


def _complementary_samples() -> list[Sample]:
    # north samples: only the text carries the signal; south samples: only the audio.
    samples = []
    for i in range(10):
        samples.append(
            Sample(
                id=f"n{i}",
                text="dialect signal north",
                audio_path="/clip/plain.wav",
                labels=DialectLabels(dialect="north"),
            )
        )
        samples.append(
            Sample(
                id=f"s{i}",
                text="just some noise here",
                audio_path="/clip/south.wav",
                labels=DialectLabels(dialect="south"),
            )
        )
    return samples


def _stubs() -> tuple[_KeyedStub, _KeyedStub]:
    return _KeyedStub(TaskType.TEXT), _KeyedStub(TaskType.AUDIO)


class TestCompareModalities:
    def test_fusion_beats_each_single_modality_when_complementary(self) -> None:
        text, audio = _stubs()
        report = compare_modalities(text, audio, _complementary_samples())
        assert report.n_samples == 20
        assert report.text.accuracy == pytest.approx(0.5)  # text only signals on the "a" half
        assert report.audio.accuracy == pytest.approx(0.5)  # audio only on the "b" half
        assert report.fused.accuracy == pytest.approx(1.0)  # fusion recovers both
        assert report.fused_beats_best_single and report.uplift_f1_macro > 0
        assert report.mcnemar_p_vs_text < 0.05  # the improvement is significant
        assert report.mcnemar_p_vs_audio < 0.05

    def test_reports_the_strategy_used(self) -> None:
        text, audio = _stubs()
        report = compare_modalities(
            text, audio, _complementary_samples(), strategy=WeightedAverageFusion((0.5, 0.5))
        )
        assert report.strategy == "weighted_average" and report.target == "dialect"

    def test_only_multimodal_labelled_samples_are_scored(self) -> None:
        text, audio = _stubs()
        samples = _complementary_samples()
        samples.append(
            Sample(id="text-only", text="signal north", labels=DialectLabels(dialect="north"))
        )
        samples.append(
            Sample(
                id="audio-only", audio_path="/clip/south.wav", labels=DialectLabels(dialect="south")
            )
        )
        report = compare_modalities(text, audio, samples)
        assert report.n_samples == 20  # the two single-modality rows are excluded

    def test_no_multimodal_samples_raises(self) -> None:
        text, audio = _stubs()
        only_text = [Sample(id="x", text="signal north", labels=DialectLabels(dialect="north"))]
        with pytest.raises(DataError, match="text, audio"):
            compare_modalities(text, audio, only_text)


class TestRendering:
    def test_markdown_byte_stable_and_complete(self) -> None:
        text, audio = _stubs()
        report = compare_modalities(text, audio, _complementary_samples())
        markdown = report.to_markdown()
        assert markdown == report.to_markdown()
        assert "# Modality comparison" in markdown
        for modality in ("text", "audio", "text+audio"):
            assert modality in markdown
        assert "McNemar" in markdown and "Fusion beats" in markdown

    def test_save_round_trips(self, tmp_path: Path) -> None:
        text, audio = _stubs()
        report = compare_modalities(text, audio, _complementary_samples())
        path = tmp_path / "comparison.json"
        report.save(path)
        reloaded = ModalityComparison.model_validate_json(path.read_text(encoding="utf-8"))
        assert reloaded.fused.accuracy == pytest.approx(1.0)
