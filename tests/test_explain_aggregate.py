"""Tests for the corpus-level dialect-evidence roll-up."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from conftest import make_samples
from tulip.cli.app import app
from tulip.core.types import DialectLabels, Explanation, Sample
from tulip.explain.aggregate import dataset_evidence
from tulip.labels.taxonomy import LabelLevel

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


# --------------------------------------------------------------- fixtures


class _FakeExplainer:
    """Return canned evidence details keyed by text, for deterministic tests."""

    def __init__(self, details_by_text: dict[str, dict[str, Any]]) -> None:
        self._details = details_by_text

    def explain(self, pipeline: Any, raw_input: Any, **kwargs: Any) -> Explanation:
        return Explanation(
            method="dialect_evidence", details=self._details.get(str(raw_input), _details())
        )


def _marker(label: str, surface: str, families: tuple[str, ...], count: int = 1) -> dict[str, Any]:
    return {
        "phenomenon": "marker",
        "label": label,
        "surface": surface,
        "families": list(families),
        "count": count,
    }


def _fired(label: str, surface: str, families: tuple[str, ...], count: int = 1) -> dict[str, Any]:
    return {
        "phenomenon": "isogloss_fired",
        "label": label,
        "surface": surface,
        "families": list(families),
        "count": count,
    }


def _details(
    *,
    markers: tuple[dict[str, Any], ...] = (),
    fired: tuple[dict[str, Any], ...] = (),
    families: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "markers": list(markers),
        "fired_rules": list(fired),
        "applicable_rules": [],
        "families": dict(families or {}),
        "caveat": "resource-defined",
    }


def _sample(sample_id: str, text: str, dialect: str, *, family: str | None = None) -> Sample:
    return Sample(
        id=sample_id,
        text=text,
        speaker_id=f"spk-{sample_id}",
        labels=DialectLabels(dialect=dialect, family=family),
        source="syn",
    )


# --------------------------------------------------------------- lift / tally


def test_class_conditional_lift_and_share() -> None:
    # A marker carried only by the 6 podhale samples, in a 50/50 corpus, is
    # twice as concentrated in podhale as the base rate: lift 2.0, share 1.0.
    explainer = _FakeExplainer(
        {"PH": _details(markers=(_marker("baca", "baca", ("lesser_polish",)),)), "SI": _details()}
    )
    samples = [_sample(f"p{i}", "PH", "podhale") for i in range(6)]
    samples += [_sample(f"s{i}", "SI", "silesia") for i in range(6)]

    report = dataset_evidence(samples, explainer=explainer)

    assert report.n_samples == 12
    assert report.n_skipped == 0
    marker = next(p for p in report.phenomena if p.label == "baca")
    assert marker.top_class == "podhale"
    assert marker.top_class_lift == pytest.approx(2.0)
    assert marker.top_class_share == pytest.approx(1.0)
    assert marker.n_samples == 6
    assert not marker.low_support
    assert report.most_diagnostic is not None
    assert report.most_diagnostic.label == "baca"


def test_document_frequency_counts_a_phenomenon_once_per_sample() -> None:
    # Two fired reflexes of the same rule in one text count once towards the
    # carrier count but twice towards the occurrence total.
    two_reflexes = _details(
        fired=(
            _fired("mazurzenie", "syc", ("masovian",)),
            _fired("mazurzenie", "zaba", ("masovian",)),
        )
    )
    explainer = _FakeExplainer({"M": two_reflexes})
    report = dataset_evidence(
        [_sample(f"m{i}", "M", "kurpie") for i in range(3)], explainer=explainer
    )

    phenomenon = report.phenomena[0]
    assert phenomenon.label == "mazurzenie"
    assert phenomenon.n_samples == 3
    assert phenomenon.total_count == 6


def test_a_low_support_phenomenon_never_headlines() -> None:
    # R is reliable (6 carriers, lift 1.667); L is a wider gap (lift 5.0) but
    # only 2 carriers, so it is flagged and ranked after the reliable R.
    explainer = _FakeExplainer(
        {
            "PH": _details(markers=(_marker("R", "r", ("lesser_polish",)),)),
            "SI": _details(markers=(_marker("L", "l", ("silesian",)),)),
            "KU": _details(),
        }
    )
    samples = [_sample(f"p{i}", "PH", "podhale") for i in range(6)]
    samples += [_sample(f"s{i}", "SI", "silesia") for i in range(2)]
    samples += [_sample(f"k{i}", "KU", "kurpie") for i in range(2)]

    report = dataset_evidence(samples, explainer=explainer)

    order = [p.label for p in report.phenomena]
    assert order.index("R") < order.index("L")
    reliable = next(p for p in report.phenomena if p.label == "R")
    low = next(p for p in report.phenomena if p.label == "L")
    assert not reliable.low_support
    assert reliable.top_class_lift == pytest.approx(10.0 / 6.0)
    assert low.low_support
    assert low.top_class_lift == pytest.approx(5.0)
    assert report.most_diagnostic is not None
    assert report.most_diagnostic.label == "R"  # the reliable one, not the wider low-support gap


def test_family_evidence_is_summed_across_the_corpus() -> None:
    explainer = _FakeExplainer(
        {
            "PH": _details(
                markers=(_marker("baca", "baca", ("lesser_polish",)),),
                families={"lesser_polish": 2},
            )
        }
    )
    report = dataset_evidence(
        [_sample(f"p{i}", "PH", "podhale") for i in range(4)], explainer=explainer
    )

    assert len(report.families) == 1
    family = report.families[0]
    assert family.family == "lesser_polish"
    assert family.n_samples == 4
    assert family.total_count == 8


def test_lift_axis_can_be_the_family_level() -> None:
    explainer = _FakeExplainer(
        {"PH": _details(markers=(_marker("baca", "baca", ("lesser_polish",)),)), "SI": _details()}
    )
    samples = [_sample(f"p{i}", "PH", "podhale", family="lesser_polish") for i in range(6)]
    samples += [_sample(f"s{i}", "SI", "silesia", family="silesian") for i in range(6)]

    report = dataset_evidence(samples, level=LabelLevel.FAMILY, explainer=explainer)

    assert report.level == "family"
    marker = next(p for p in report.phenomena if p.label == "baca")
    assert marker.top_class == "lesser_polish"


def test_samples_without_text_or_a_gold_label_are_skipped() -> None:
    explainer = _FakeExplainer(
        {"PH": _details(markers=(_marker("baca", "baca", ("lesser_polish",)),))}
    )
    labelled = _sample("p0", "PH", "podhale")
    unlabelled = Sample(id="u", text="PH", speaker_id="s", labels=DialectLabels(), source="syn")
    audio_only = Sample(
        id="a",
        audio_path="clip.wav",  # type: ignore[arg-type]
        speaker_id="s",
        labels=DialectLabels(dialect="podhale"),
        source="syn",
    )
    report = dataset_evidence([labelled, unlabelled, audio_only], explainer=explainer)

    assert report.n_samples == 1
    assert report.n_skipped == 2


def test_blank_text_is_skipped_not_crashed() -> None:
    # A Sample is valid with empty or whitespace-only text; the real explainer
    # rejects such input, so the roll-up must skip the row rather than abort.
    from tulip.explain.dialect_evidence import DialectEvidenceExplainer

    samples = [
        Sample(
            id="e", text="", speaker_id="s", labels=DialectLabels(dialect="podhale"), source="syn"
        ),
        Sample(
            id="w",
            text="   ",
            speaker_id="s",
            labels=DialectLabels(dialect="podhale"),
            source="syn",
        ),
    ]
    report = dataset_evidence(samples, explainer=DialectEvidenceExplainer())

    assert report.n_samples == 0
    assert report.n_skipped == 2
    assert report.phenomena == ()


def test_report_is_byte_stable(tmp_path: Path) -> None:
    explainer = _FakeExplainer(
        {"PH": _details(markers=(_marker("baca", "baca", ("lesser_polish",)),)), "SI": _details()}
    )
    samples = [_sample(f"p{i}", "PH", "podhale") for i in range(6)]
    samples += [_sample(f"s{i}", "SI", "silesia") for i in range(6)]

    first = dataset_evidence(samples, explainer=explainer)
    second = dataset_evidence(samples, explainer=explainer)
    assert first.model_dump() == second.model_dump()

    first.save(tmp_path / "a.json")
    second.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_empty_corpus_has_no_headline() -> None:
    report = dataset_evidence([], explainer=_FakeExplainer({}))
    assert report.n_samples == 0
    assert report.phenomena == ()
    assert report.families == ()
    assert report.most_diagnostic is None
    assert "none with adequate support" in report.to_markdown()


def test_markdown_names_the_sections_and_marks_low_support() -> None:
    explainer = _FakeExplainer({"SI": _details(markers=(_marker("L", "l", ("silesian",)),))})
    report = dataset_evidence([_sample("s0", "SI", "silesia")], explainer=explainer)
    markdown = report.to_markdown()
    assert "Dialect evidence" in markdown
    assert "Phenomena by class-conditional lift" in markdown
    assert "Family evidence" in markdown
    assert "L *" in markdown  # the single-carrier phenomenon is low-support


def test_aggregate_import_pulls_no_neural_deps() -> None:
    # sklearn is a base dependency of the explain package; torch/shap/lime are
    # heavy optionals the roll-up must never trigger.
    code = (
        "import sys, tulip.explain.aggregate as _;"
        "heavy=[m for m in ('torch', 'shap', 'lime') if m in sys.modules];"
        "raise SystemExit(1 if heavy else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)  # noqa: S603  (trusted, fixed input)
    assert result.returncode == 0


# --------------------------------------------------------------- CLI


def _write_corpus(path: Path) -> None:
    samples = make_samples(repeats=4)
    path.write_text(
        "\n".join(sample.model_dump_json() for sample in samples) + "\n", encoding="utf-8"
    )


def test_explain_global_command_runs(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    result = runner.invoke(app, ["explain-global", str(corpus)])
    assert result.exit_code == 0, result.output
    assert "Dialect evidence" in result.output


def test_explain_global_json(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    result = runner.invoke(app, ["explain-global", str(corpus), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "phenomena" in payload
    assert "families" in payload


def test_explain_global_rejects_an_unknown_level(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    result = runner.invoke(app, ["explain-global", str(corpus), "--level", "nonsense"])
    assert result.exit_code == 1
    assert "error:" in result.output
