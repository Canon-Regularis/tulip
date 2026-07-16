"""Tests for teacher -> student knowledge distillation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.core.exceptions import DataError
from tulip.core.types import DialectLabels, Sample, TaskType
from tulip.labels.taxonomy import LabelLevel
from tulip.pipeline import DialectClassifier, DistillationConfig, distill

if TYPE_CHECKING:
    from pathlib import Path


def _sample(sid: str, text: str, dialect: str, speaker: str) -> Sample:
    return Sample(
        id=sid, text=text, speaker_id=speaker, labels=DialectLabels(dialect=dialect), source="test"
    )


def _labelled() -> list[Sample]:
    return [
        _sample("k0", "psiwo warzą jesce po staremu bór", "kurpie", "k0"),
        _sample("k1", "psiwo warzą jesce nase pole rano", "kurpie", "k1"),
        _sample("k2", "psiwo warzą jesce stary las wieczór", "kurpie", "k2"),
        _sample("p0", "hej baca się pyto kaj owce pasą holi", "podhale", "p0"),
        _sample("p1", "hej baca się pyto kaj hole nase grań", "podhale", "p1"),
        _sample("p2", "hej baca się pyto kaj wysoko granie", "podhale", "p2"),
    ]


@pytest.fixture
def teacher() -> DialectClassifier:
    model = DialectClassifier(
        model="logistic_regression",
        features=["char_tfidf"],
        task=TaskType.TEXT,
        target=LabelLevel.DIALECT,
        seed=42,
    )
    return model.fit(_labelled())


@pytest.fixture
def transfer() -> list[Sample]:
    # A pool the teacher labels; any labels on it are ignored by distillation.
    return [
        _sample("t0", "psiwo warzą po staremu w borze", "kurpie", "t0"),
        _sample("t1", "psiwo warzą jesce nase stare pole", "kurpie", "t1"),
        _sample("t2", "hej baca kaj owce pasą na holi", "podhale", "t2"),
        _sample("t3", "hej baca kaj hole wysoko granie", "podhale", "t3"),
    ]


@pytest.fixture
def test_set() -> list[Sample]:
    return [
        _sample("e0", "psiwo warzą jesce po staremu", "kurpie", "e0"),
        _sample("e1", "hej baca się pyto kaj owce", "podhale", "e1"),
    ]


def test_distills_into_a_student(teacher, transfer, test_set) -> None:
    report = distill(
        teacher=teacher,
        transfer=transfer,
        test=test_set,
        student_model="naive_bayes",
        features=["char_tfidf"],
        measure_cost=False,
    )
    assert report.teacher_model == "logistic_regression"
    assert report.student_model == "naive_bayes"
    assert report.target == "dialect"
    assert report.n_transfer == 4
    assert 1 <= report.n_transfer_used <= 4
    assert 0.0 <= report.student_accuracy <= 1.0
    assert 0.0 <= report.agreement <= 1.0
    if report.teacher_accuracy > 0.0:
        assert report.retention == pytest.approx(report.student_accuracy / report.teacher_accuracy)
    assert report.teacher_efficiency is None  # measure_cost=False


def test_confidence_filter_gates_transfer(teacher, transfer, test_set) -> None:
    # At confidence 0.0 every teacher label is kept; at 1.0 none survive (a
    # logistic model's probability is never exactly 1.0), so the gate spans the
    # full range and the student has nothing to learn from at the top.
    lenient = distill(
        teacher=teacher,
        transfer=transfer,
        test=test_set,
        student_model="naive_bayes",
        features=["char_tfidf"],
        config=DistillationConfig(min_teacher_confidence=0.0),
        measure_cost=False,
    )
    assert lenient.n_transfer_used == 4

    with pytest.raises(DataError, match="nothing to train the student on"):
        distill(
            teacher=teacher,
            transfer=transfer,
            test=test_set,
            student_model="naive_bayes",
            features=["char_tfidf"],
            config=DistillationConfig(min_teacher_confidence=1.0),
            measure_cost=False,
        )


def test_no_usable_transfer_raises(teacher, test_set) -> None:
    with pytest.raises(DataError, match="nothing to train the student on"):
        distill(
            teacher=teacher,
            transfer=[],
            test=test_set,
            student_model="naive_bayes",
            features=["char_tfidf"],
            measure_cost=False,
        )


def test_report_is_deterministic_without_cost(teacher, transfer, test_set, tmp_path: Path) -> None:
    first = distill(
        teacher=teacher,
        transfer=transfer,
        test=test_set,
        student_model="naive_bayes",
        features=["char_tfidf"],
        measure_cost=False,
    )
    second = distill(
        teacher=teacher,
        transfer=transfer,
        test=test_set,
        student_model="naive_bayes",
        features=["char_tfidf"],
        measure_cost=False,
    )
    assert first.model_dump() == second.model_dump()
    first.save(tmp_path / "a.json")
    second.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_cost_is_measured_with_a_workdir(teacher, transfer, test_set, tmp_path: Path) -> None:
    report = distill(
        teacher=teacher,
        transfer=transfer,
        test=test_set,
        student_model="naive_bayes",
        features=["char_tfidf"],
        measure_cost=True,
        workdir=tmp_path / "models",
    )
    assert report.teacher_efficiency is not None
    assert report.student_efficiency is not None
    # A workdir was given, so on-disk size is measured for both.
    assert report.teacher_efficiency.model_size_bytes is not None
    assert report.student_efficiency.model_size_bytes is not None


def test_cost_measurement_tolerates_a_modality_less_test_sample(
    teacher, transfer, tmp_path: Path
) -> None:
    # measure_efficiency times predict_samples, which is stricter than the
    # accuracy path; a text-less (audio-only) sample must not crash the run.
    heterogeneous = [
        _sample("e0", "psiwo warzą jesce po staremu", "kurpie", "e0"),
        Sample(
            id="audio-only",
            text=None,
            audio_path=tmp_path / "clip.wav",
            speaker_id="a0",
            labels=DialectLabels(dialect="podhale"),
            source="test",
        ),
    ]
    report = distill(
        teacher=teacher,
        transfer=transfer,
        test=heterogeneous,
        student_model="naive_bayes",
        features=["char_tfidf"],
        measure_cost=True,
    )
    assert report.teacher_efficiency is not None
    assert report.teacher_efficiency.n_samples == 1  # timed only the text sample


def test_markdown_names_both_models(teacher, transfer, test_set) -> None:
    markdown = distill(
        teacher=teacher,
        transfer=transfer,
        test=test_set,
        student_model="naive_bayes",
        features=["char_tfidf"],
        measure_cost=False,
    ).to_markdown()
    assert "Distillation" in markdown
    assert "logistic_regression" in markdown and "naive_bayes" in markdown


def test_cli_distill(tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from tulip.cli.app import app
    from tulip.utils.io import write_jsonl

    teacher = DialectClassifier(
        model="logistic_regression",
        features=["char_tfidf"],
        task=TaskType.TEXT,
        target=LabelLevel.DIALECT,
        seed=42,
    ).fit(_labelled())
    teacher_dir = teacher.save(tmp_path / "teacher")

    transfer = tmp_path / "transfer.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(
        transfer,
        [
            {"id": "t0", "text": "psiwo warzą po staremu", "speaker_id": "t0", "dialect": "kurpie"},
            {
                "id": "t1",
                "text": "hej baca kaj owce pasą",
                "speaker_id": "t1",
                "dialect": "podhale",
            },
        ],
    )
    write_jsonl(
        test,
        [
            {"id": "e0", "text": "psiwo warzą jesce", "speaker_id": "e0", "dialect": "kurpie"},
            {"id": "e1", "text": "hej baca kaj owce", "speaker_id": "e1", "dialect": "podhale"},
        ],
    )

    # No --feature: a classical student defaults to char_tfidf, so the bare
    # command works out of the box.
    result = CliRunner().invoke(
        app,
        ["distill", str(teacher_dir), str(transfer), str(test), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["teacher_model"] == "logistic_regression"
    assert payload["student_model"] == "logistic_regression"
