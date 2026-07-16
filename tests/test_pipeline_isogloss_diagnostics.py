"""Tests for per-isogloss diagnostic evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.core.types import DialectLabels, Sample, TaskType
from tulip.labels.taxonomy import LabelLevel
from tulip.pipeline import DialectClassifier, isogloss_diagnostics

if TYPE_CHECKING:
    from pathlib import Path


def _sample(sid: str, text: str, dialect: str, speaker: str) -> Sample:
    return Sample(
        id=sid,
        text=text,
        speaker_id=speaker,
        labels=DialectLabels(dialect=dialect),
        source="test",
    )


@pytest.fixture
def classifier() -> DialectClassifier:
    train = [
        _sample("k0", "psiwo warzą jesce po staremu", "kurpie", "k-spk0"),
        _sample("k1", "psiwo warzą jesce nase pole", "kurpie", "k-spk1"),
        _sample("k2", "psiwo warzą jesce stary bór", "kurpie", "k-spk2"),
        _sample("p0", "hej baca się pyto kaj owce pasą", "podhale", "p-spk0"),
        _sample("p1", "hej baca się pyto kaj hole nase", "podhale", "p-spk1"),
        _sample("p2", "hej baca się pyto kaj grań wysoko", "podhale", "p-spk2"),
    ]
    model = DialectClassifier(
        model="logistic_regression",
        features=["char_tfidf"],
        task=TaskType.TEXT,
        target=LabelLevel.DIALECT,
        seed=42,
    )
    return model.fit(train)


@pytest.fixture
def test_samples() -> list[Sample]:
    # kurpie: three with the soft-labial reflex (psiwo), two standard (piwo);
    # plus two podhale, which the soft_labials diagnostic must exclude.
    return [
        _sample("tk0", "psiwo warzą dobre dzis", "kurpie", "tk-spk0"),
        _sample("tk1", "psiwo warzą dobre rano", "kurpie", "tk-spk1"),
        _sample("tk2", "psiwo warzą dobre wieczór", "kurpie", "tk-spk2"),
        _sample("tk3", "piwo warzą dobre dzis", "kurpie", "tk-spk3"),
        _sample("tk4", "piwo warzą dobre rano", "kurpie", "tk-spk4"),
        _sample("tp0", "hej baca kaj owce pasą", "podhale", "tp-spk0"),
        _sample("tp1", "hej baca kaj hole nase", "podhale", "tp-spk1"),
    ]


def _find(report, rule: str):
    return next(d for d in report.diagnostics if d.rule == rule)


def test_only_detectable_rules_are_diagnosed(
    classifier: DialectClassifier, test_samples: list[Sample]
) -> None:
    report = isogloss_diagnostics(classifier, test_samples)
    names = {d.rule for d in report.diagnostics}
    assert "soft_labials" in names  # detectable
    # Mergers cannot be positively detected, so they are not diagnosed.
    assert names.isdisjoint({"mazurzenie", "kaszubienie", "silesian_final_ch"})


def test_present_absent_split_and_scope(
    classifier: DialectClassifier, test_samples: list[Sample]
) -> None:
    report = isogloss_diagnostics(classifier, test_samples, min_support=2)
    soft = _find(report, "soft_labials")

    assert soft.dialects == ("kurpie",)
    assert soft.n_present == 3  # the psiwo samples
    assert soft.n_absent == 2  # the piwo samples
    # The two podhale samples are out of scope, so they inflate neither group.
    assert soft.n_present + soft.n_absent == 5
    assert soft.accuracy_present is not None and 0.0 <= soft.accuracy_present <= 1.0
    assert soft.accuracy_absent is not None and 0.0 <= soft.accuracy_absent <= 1.0
    assert soft.delta == pytest.approx(soft.accuracy_present - soft.accuracy_absent)
    assert soft.low_support is False  # both groups >= min_support (2)


def test_low_support_flag_tracks_min_support(
    classifier: DialectClassifier, test_samples: list[Sample]
) -> None:
    report = isogloss_diagnostics(classifier, test_samples)  # default min_support=5
    soft = _find(report, "soft_labials")
    assert soft.low_support is True  # 3 and 2 are both under 5


def test_report_is_deterministic(
    classifier: DialectClassifier, test_samples: list[Sample], tmp_path: Path
) -> None:
    first = isogloss_diagnostics(classifier, test_samples)
    second = isogloss_diagnostics(classifier, test_samples)
    assert first.model_dump() == second.model_dump()
    first.save(tmp_path / "a.json")
    second.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_markdown_names_the_report(
    classifier: DialectClassifier, test_samples: list[Sample]
) -> None:
    markdown = isogloss_diagnostics(classifier, test_samples).to_markdown()
    assert "Isogloss diagnostics" in markdown
    assert "soft_labials" in markdown


def test_custom_rules_path_is_honoured(
    classifier: DialectClassifier, test_samples: list[Sample], tmp_path: Path
) -> None:
    rules = tmp_path / "rules.yaml"
    # A detectable rule whose reflex "kaj" fires on the podhale samples.
    rules.write_text(
        "version: 1\nrules:\n"
        "  - name: podhale_kaj\n"
        "    dialects: [podhale]\n"
        "    detectable: true\n"
        "    where: anywhere\n"
        "    map:\n"
        "      kai: kaj\n",
        encoding="utf-8",
    )
    report = isogloss_diagnostics(classifier, test_samples, rules_path=rules)
    names = {d.rule for d in report.diagnostics}
    assert names == {"podhale_kaj"}  # the bundled rules are replaced
    podhale = _find(report, "podhale_kaj")
    assert podhale.n_present == 2  # both podhale test samples carry "kaj"


def test_cli_isogloss_diagnostics(tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from tulip.cli.app import app

    train = [
        _sample("k0", "psiwo warzą jesce po staremu", "kurpie", "k-spk0"),
        _sample("k1", "psiwo warzą jesce nase pole", "kurpie", "k-spk1"),
        _sample("p0", "hej baca się pyto kaj owce pasą", "podhale", "p-spk0"),
        _sample("p1", "hej baca się pyto kaj hole nase", "podhale", "p-spk1"),
    ]
    model = DialectClassifier(
        model="logistic_regression",
        features=["char_tfidf"],
        task=TaskType.TEXT,
        target=LabelLevel.DIALECT,
        seed=42,
    ).fit(train)
    model_dir = model.save(tmp_path / "model")

    # Reuse the manifest reader by writing the test samples as a JSONL split.
    from tulip.utils.io import write_jsonl

    data = tmp_path / "test.jsonl"
    write_jsonl(
        data,
        [
            {"id": "tk0", "text": "psiwo warzą dobre", "speaker_id": "s0", "dialect": "kurpie"},
            {"id": "tk1", "text": "piwo warzą dobre", "speaker_id": "s1", "dialect": "kurpie"},
        ],
    )

    result = CliRunner().invoke(
        app, ["isogloss-diagnostics", str(model_dir), str(data), "--min-support", "1", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert any(d["rule"] == "soft_labials" for d in payload["diagnostics"])
