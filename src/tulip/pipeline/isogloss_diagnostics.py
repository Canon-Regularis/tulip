"""Per-isogloss diagnostics: does accuracy collapse when a marker is absent?

A dialect classifier can look strong on aggregate yet be leaning on a single
surface cue. This asks the dialectology-native question directly: for each
detectable isogloss (a group-defining sound change, e.g. mazurzenie), take the
test samples of the dialects that isogloss signals, split them by whether the
dialectal reflex is actually present in the text, and compare accuracy on the
two halves. A large positive gap (accurate when the marker fired, wrong when it
did not) means the model reads the marker rather than the dialect.

It reuses two existing blocks: the bidirectional rule engine
(:func:`~tulip.features.text.phonological_rules.load_phonological_rules`) detects
each marker per sample, and :func:`~tulip.pipeline.experiment.collect_predictions`
supplies the per-sample correctness the error-analysis layer already builds on.
Only *detectable* rules are diagnosed: a merger's reflex is ordinary standard
Polish, so its presence cannot be read off the surface. This is a text
diagnostic; audio-only samples (no text) are skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tulip._serialize import round_floats, write_sorted_json
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.core.types import Sample
    from tulip.pipeline.classifier import DialectClassifier

__all__ = [
    "ISOGLOSS_FLOAT_DIGITS",
    "IsoglossDiagnostic",
    "IsoglossReport",
    "isogloss_diagnostics",
]

_logger = get_logger(__name__)

#: Group (present or absent) smaller than this is flagged low-support: its
#: accuracy is noisy and the gap must not be read as a headline finding.
DEFAULT_MIN_SUPPORT = 5

#: Stored floats are rounded to this many digits so a saved report is
#: byte-identical when the content is, matching the other rigor reports.
ISOGLOSS_FLOAT_DIGITS = 6


class IsoglossDiagnostic(BaseModel):
    """One isogloss: accuracy with the marker present versus absent."""

    model_config = ConfigDict(frozen=True)

    rule: str
    dialects: tuple[str, ...]
    n_present: int = Field(ge=0)
    accuracy_present: float | None = Field(default=None, ge=0.0, le=1.0)
    n_absent: int = Field(ge=0)
    accuracy_absent: float | None = Field(default=None, ge=0.0, le=1.0)
    delta: float | None = None
    low_support: bool = False


class IsoglossReport(BaseModel):
    """Per-isogloss diagnostics for one model on one split, largest gap first."""

    model_config = ConfigDict(frozen=True)

    model: str
    split: str
    n_samples: int = Field(ge=0)
    diagnostics: tuple[IsoglossDiagnostic, ...]

    def to_markdown(self) -> str:
        """Render the diagnostics as a markdown table (low-support marked *)."""
        from tulip.evaluation._format import format_metric, markdown_table

        def cell(value: float | None) -> str:
            return "n/a" if value is None else format_metric(value)

        rows = [
            (
                diagnostic.rule + (" *" if diagnostic.low_support else ""),
                ", ".join(diagnostic.dialects) or "(any)",
                str(diagnostic.n_present),
                cell(diagnostic.accuracy_present),
                str(diagnostic.n_absent),
                cell(diagnostic.accuracy_absent),
                cell(diagnostic.delta),
            )
            for diagnostic in self.diagnostics
        ] or [("n/a", "n/a", "0", "n/a", "0", "n/a", "n/a")]
        title = f"# Isogloss diagnostics: {self.model} ({self.split})"
        headers = (
            "Isogloss",
            "Dialects",
            "N present",
            "Acc present",
            "N absent",
            "Acc absent",
            "Δ",
        )
        return f"{title}\n\n{markdown_table(headers, rows)}"

    def save(self, path: Path | str) -> None:
        """Write the report as deterministic JSON (sorted keys, rounded floats)."""
        payload = round_floats(self.model_dump(mode="json"), ISOGLOSS_FLOAT_DIGITS)
        write_sorted_json(Path(path), payload)


def isogloss_diagnostics(
    classifier: DialectClassifier,
    samples: Sequence[Sample],
    *,
    rules_path: str | Path | None = None,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> IsoglossReport:
    """Score a fitted classifier's accuracy with versus without each isogloss.

    For every detectable rule, the samples whose gold label is one the rule
    signals (its ``dialects``/``families``; all samples when the rule names
    none) are split by whether the dialectal reflex fired in the text, and
    accuracy is compared across the split.

    Args:
        classifier: A fitted classifier; scored once via
            :func:`~tulip.pipeline.experiment.collect_predictions`.
        samples: The labelled evaluation samples (must carry text to read markers).
        rules_path: Phonological rule file replacing the bundled set; ``None``
            uses the bundled one.
        min_support: Group size below which a diagnostic is flagged low-support.

    Returns:
        An :class:`IsoglossReport`, largest present-minus-absent gap first.

    Raises:
        DataError: if no sample is evaluable at the classifier's target level
            (raised by :func:`collect_predictions`).
    """
    from tulip.features.text._tokenize import word_tokens
    from tulip.features.text.phonological_rules import load_phonological_rules
    from tulip.pipeline.experiment import collect_predictions

    predictions = collect_predictions(classifier, samples)
    by_id = {sample.id: sample for sample in samples}
    rules = [rule for rule in load_phonological_rules(rules_path) if rule.detectable]

    diagnostics: list[IsoglossDiagnostic] = []
    for rule in rules:
        scope = {value.lower() for value in (*rule.dialects, *rule.families)}
        present: list[bool] = []
        absent: list[bool] = []
        for record in predictions.records:
            if scope and record.y_true.lower() not in scope:
                continue
            sample = by_id.get(record.id)
            if sample is None or sample.text is None:
                continue
            tokens = word_tokens(str(sample.text), lowercase=True)
            fired = any(rule.fired_matches(token) for token in tokens)
            (present if fired else absent).append(record.correct)

        acc_present = _accuracy(present)
        acc_absent = _accuracy(absent)
        delta = None if acc_present is None or acc_absent is None else acc_present - acc_absent
        diagnostics.append(
            IsoglossDiagnostic(
                rule=rule.name,
                dialects=rule.dialects,
                n_present=len(present),
                accuracy_present=acc_present,
                n_absent=len(absent),
                accuracy_absent=acc_absent,
                delta=delta,
                low_support=len(present) < min_support or len(absent) < min_support,
            )
        )

    # Largest reliance first; diagnostics with no computable gap sort last, ties
    # broken by rule name for a total, reproducible order.
    diagnostics.sort(key=lambda d: (d.delta is None, -(d.delta or 0.0), d.rule))
    _logger.info(
        "isogloss diagnostics: %d detectable rules over %d samples",
        len(diagnostics),
        len(predictions),
    )
    return IsoglossReport(
        model=predictions.model,
        split=predictions.split,
        n_samples=len(predictions),
        diagnostics=tuple(diagnostics),
    )


def _accuracy(flags: Sequence[bool]) -> float | None:
    """Fraction correct over a group, or ``None`` for an empty group."""
    if not flags:
        return None
    return sum(1 for flag in flags if flag) / len(flags)
