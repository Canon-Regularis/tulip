"""Knowledge distillation: a large teacher trains a small, fast student.

A transformer or LLM baseline may win on accuracy while being a hundred times
larger and slower than a classical model. Distillation asks what a cheap model
can recover of that win: the teacher labels a transfer pool, a small student
trains on those labels, and the report puts the student's accuracy next to the
teacher's alongside the size and latency it costs, which is the size-versus
accuracy story the efficiency machinery already tells for a single model.

This is *hard-label* distillation: the student learns the teacher's argmax, and
the teacher's probability is used only to gate which labels transfer (a low
``min_teacher_confidence`` keeps them all, a high one keeps the confident ones).
A classical student cannot consume soft targets without soft-label plumbing the
whole feature pipeline does not have, so this stays a cheap reuse of the
existing training path rather than pretending otherwise.

Everything reuses existing blocks: the student is a
:class:`~tulip.pipeline.classifier.DialectClassifier`, accuracy comes from
:func:`~tulip.pipeline.experiment.evaluate_samples`, and the cost numbers reuse
:func:`~tulip.evaluation.efficiency.measure_efficiency`. The accuracy fields are
deterministic under a fixed seed; the efficiency block is machine dependent
(latency), exactly like ``efficiency.json``, so a saved report is not byte
stable when costs are measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import round_floats, write_sorted_json
from tulip.core.exceptions import DataError
from tulip.evaluation.efficiency import EfficiencyRecord
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.core.types import Sample, TaskType
    from tulip.pipeline.classifier import ComponentLike, DialectClassifier

__all__ = [
    "DISTILL_FLOAT_DIGITS",
    "DistillationConfig",
    "DistillationReport",
    "distill",
]

_logger = get_logger(__name__)

#: Stored floats are rounded to this many digits so the deterministic fields of
#: a saved report are byte-identical when the content is (the efficiency block,
#: when present, is machine dependent regardless).
DISTILL_FLOAT_DIGITS = 6


class DistillationConfig(BaseModel):
    """Parameters governing one distillation run.

    A standalone, module-owned schema, not an extension of the frozen
    :class:`~tulip.config.schemas.ExperimentConfig`.

    Attributes:
        min_teacher_confidence: Teacher predictions below this top-class
            probability are dropped from the transfer set (``0.0`` keeps every
            label). The student trains only on the labels that survive.
        seed: Seed applied before the student fit, making the run reproducible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_teacher_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int = 42


class DistillationReport(BaseModel):
    """How much of the teacher a distilled student keeps, and at what cost."""

    model_config = ConfigDict(frozen=True)

    teacher_model: str
    student_model: str
    target: str
    n_transfer: int = Field(ge=0)
    n_transfer_used: int = Field(ge=0)
    teacher_accuracy: float = Field(ge=0.0, le=1.0)
    student_accuracy: float = Field(ge=0.0, le=1.0)
    retention: float | None = None
    agreement: float = Field(ge=0.0, le=1.0)
    teacher_efficiency: EfficiencyRecord | None = None
    student_efficiency: EfficiencyRecord | None = None

    def to_markdown(self) -> str:
        """Render accuracy retention and (when measured) the cost comparison."""
        from tulip.evaluation._format import format_metric, markdown_table

        retention = "n/a" if self.retention is None else format_metric(self.retention)
        rows = [
            ("Teacher accuracy", format_metric(self.teacher_accuracy)),
            ("Student accuracy", format_metric(self.student_accuracy)),
            ("Retention (student/teacher)", retention),
            ("Student-teacher agreement", format_metric(self.agreement)),
            ("Transfer labels used", f"{self.n_transfer_used}/{self.n_transfer}"),
        ]
        parts = [
            f"# Distillation: {self.teacher_model} -> {self.student_model} ({self.target})",
            markdown_table(("Metric", "Value"), rows),
        ]
        if self.teacher_efficiency is not None and self.student_efficiency is not None:
            parts.append("## Cost (machine dependent)")
            parts.append(
                markdown_table(
                    ("Model", "Latency (ms)", "Params", "Size (bytes)"),
                    [
                        _efficiency_row("teacher", self.teacher_efficiency),
                        _efficiency_row("student", self.student_efficiency),
                    ],
                )
            )
        return "\n\n".join(parts)

    def save(self, path: Path | str) -> None:
        """Write the report as JSON (sorted keys, rounded floats)."""
        payload = round_floats(self.model_dump(mode="json"), DISTILL_FLOAT_DIGITS)
        write_sorted_json(Path(path), payload)


def _efficiency_row(role: str, record: EfficiencyRecord) -> tuple[str, str, str, str]:
    """One markdown row for a model's measured cost."""
    return (
        role,
        f"{record.latency_ms:.4f}",
        "n/a" if record.n_params is None else str(record.n_params),
        "n/a" if record.model_size_bytes is None else str(record.model_size_bytes),
    )


def distill(
    *,
    teacher: DialectClassifier,
    transfer: Sequence[Sample],
    test: Sequence[Sample],
    student_model: ComponentLike,
    features: Sequence[ComponentLike] = (),
    config: DistillationConfig | None = None,
    measure_cost: bool = True,
    workdir: Path | str | None = None,
) -> DistillationReport:
    """Distil a fitted teacher into a small student and report the trade-off.

    The teacher labels ``transfer`` (its own gold labels, if any, are ignored),
    the confident labels train the student, and both are scored on ``test``. The
    student inherits the teacher's task and target, so the two share a label
    space.

    Args:
        teacher: A fitted classifier producing the transfer labels.
        transfer: The pool the teacher labels; labels on it are not used.
        test: Gold-labelled samples both models are scored on.
        student_model: The small model to distil into (registry name, mapping,
            or component config).
        features: Feature components for the student (empty for a raw-input model).
        config: Distillation parameters; defaults to :class:`DistillationConfig`.
        measure_cost: Also measure latency, parameter count, and (with
            ``workdir``) on-disk size for both models.
        workdir: Directory to save both models under so their on-disk size is
            measured; ``None`` leaves sizes unset.

    Returns:
        A :class:`DistillationReport`.

    Raises:
        DataError: if no transfer label clears ``min_teacher_confidence`` (the
            student would have nothing to learn from), or ``test`` has nothing
            evaluable.
    """
    config = config or DistillationConfig()

    from tulip.pipeline.classifier import DialectClassifier
    from tulip.pipeline.experiment import evaluate_samples
    from tulip.utils.seed import set_global_seed

    task = teacher.task
    target = teacher.target

    labelled = _teacher_labels(teacher, transfer, task, config.min_teacher_confidence)
    if not labelled:
        raise DataError(
            f"distillation: no teacher label cleared min_teacher_confidence="
            f"{config.min_teacher_confidence}; nothing to train the student on"
        )

    set_global_seed(config.seed)
    student = DialectClassifier(
        model=student_model, features=features, task=task, target=target, seed=config.seed
    )
    student.fit(labelled)

    teacher_report = evaluate_samples(teacher, test, name="distill-teacher")
    student_report = evaluate_samples(student, test, name="distill-student")
    agreement = _agreement(teacher, student, test, task)

    teacher_efficiency: EfficiencyRecord | None = None
    student_efficiency: EfficiencyRecord | None = None
    if measure_cost:
        teacher_efficiency, student_efficiency = _measure_costs(
            teacher, student, test, workdir=workdir
        )

    teacher_accuracy = teacher_report.accuracy
    retention = student_report.accuracy / teacher_accuracy if teacher_accuracy > 0.0 else None
    _logger.info(
        "distillation: student %s keeps %.1f%% of teacher %s accuracy",
        student.model_config.name,
        100.0 * (retention or 0.0),
        teacher.model_config.name,
    )
    return DistillationReport(
        teacher_model=teacher.model_config.name,
        student_model=student.model_config.name,
        target=target.value,
        n_transfer=len(transfer),
        n_transfer_used=len(labelled),
        teacher_accuracy=teacher_accuracy,
        student_accuracy=student_report.accuracy,
        retention=retention,
        agreement=agreement,
        teacher_efficiency=teacher_efficiency,
        student_efficiency=student_efficiency,
    )


def _teacher_labels(
    teacher: DialectClassifier, transfer: Sequence[Sample], task: TaskType, min_confidence: float
) -> list[Sample]:
    """Label the transfer pool with the teacher, keeping confident predictions.

    Each surviving sample is a fresh copy carrying the teacher's label at the
    teacher's target level and a provenance marker, so distilled labels stay
    separable from gold ones.
    """
    from tulip.core.types import DialectLabels
    from tulip.pipeline._assembly import raw_input_of

    candidates: list[tuple[Sample, object]] = []
    for sample in transfer:
        raw = raw_input_of(sample, task)
        if raw is not None:
            candidates.append((sample, raw))
    if not candidates:
        return []

    predictions = teacher.predict_batch([raw for _, raw in candidates])
    target = teacher.target
    labelled: list[Sample] = []
    for (sample, _), prediction in zip(candidates, predictions, strict=True):
        if prediction.abstained or prediction.label is None:
            continue
        if prediction.confidence < min_confidence:
            continue
        labels = DialectLabels(**{target.value: prediction.label})
        metadata = {
            **sample.metadata,
            "distilled_from": teacher.model_config.name,
            "teacher_confidence": prediction.confidence,
        }
        labelled.append(sample.model_copy(update={"labels": labels, "metadata": metadata}))
    return labelled


def _agreement(
    teacher: DialectClassifier,
    student: DialectClassifier,
    test: Sequence[Sample],
    task: TaskType,
) -> float:
    """Fraction of test inputs on which the student's argmax matches the teacher's."""
    from tulip.pipeline._assembly import raw_input_of

    raws = [raw for sample in test if (raw := raw_input_of(sample, task)) is not None]
    if not raws:
        return 0.0
    teacher_preds = teacher.predict_batch(raws)
    student_preds = student.predict_batch(raws)
    matches = sum(
        1
        for teacher_pred, student_pred in zip(teacher_preds, student_preds, strict=True)
        if teacher_pred.label == student_pred.label
    )
    return matches / len(raws)


def _measure_costs(
    teacher: DialectClassifier,
    student: DialectClassifier,
    test: Sequence[Sample],
    *,
    workdir: Path | str | None,
) -> tuple[EfficiencyRecord, EfficiencyRecord]:
    """Measure latency, parameters, and (with a workdir) on-disk size for both.

    ``measure_efficiency`` times ``predict_samples``, which (unlike the tolerant
    ``evaluate_samples``) raises on a sample lacking the task's input modality.
    A distillation test set may be heterogeneous, so latency is timed over only
    the modality-bearing samples; the accuracy path already skipped the rest.
    """
    from tulip.evaluation.efficiency import measure_efficiency
    from tulip.pipeline._assembly import raw_input_of

    timed = [sample for sample in test if raw_input_of(sample, teacher.task) is not None]
    teacher_dir = student_dir = None
    if workdir is not None:
        base = Path(workdir)
        teacher_dir = teacher.save(base / "teacher")
        student_dir = student.save(base / "student")
    teacher_efficiency = measure_efficiency(
        teacher, timed, model=teacher.model_config.name, model_dir=teacher_dir
    )
    student_efficiency = measure_efficiency(
        student, timed, model=student.model_config.name, model_dir=student_dir
    )
    return teacher_efficiency, student_efficiency
