"""Tests for contrastive dialect analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.core.types import DialectLabels, Sample
from tulip.explain.contrast import ContrastReport, contrast_dialects
from tulip.labels.taxonomy import LabelLevel

if TYPE_CHECKING:
    from pathlib import Path


def _samples(dialect: str, texts: list[str], *, level_family: str | None = None) -> list[Sample]:
    labels = DialectLabels(family=level_family) if level_family else DialectLabels(dialect=dialect)
    return [Sample(id=f"{dialect}-{i}", text=text, labels=labels) for i, text in enumerate(texts)]


def _podhale_vs_spisz() -> list[Sample]:
    # Podhale docs carry the lexicon marker "baca"; spisz docs do not. Each group
    # also has its own distinctive word-final endings.
    a = _samples("podhale", [f"hej baca sie pyto kaj owce pasom na holi {i}" for i in range(15)])
    b = _samples("spisz", [f"dzien dobry jak sie dzisiaj czujesz sasiedzie {i}" for i in range(15)])
    return a + b


class TestContrastDialects:
    def test_surfaces_a_distinguishing_lexical_marker(self) -> None:
        report = contrast_dialects(_podhale_vs_spisz(), "podhale", "spisz", min_support=3)
        assert report.n_docs_a == 15 and report.n_docs_b == 15
        baca = next(f for f in report.features if f.feature == "baca")
        assert baca.category == "lexical"
        assert baca.favored == "podhale" and baca.log_odds > 0
        assert baca.rate_a == pytest.approx(1.0) and baca.rate_b == pytest.approx(0.0)
        assert baca.significant  # present in all A, none of B -> strongly significant

    def test_features_sorted_by_effect_size(self) -> None:
        features = contrast_dialects(
            _podhale_vs_spisz(), "podhale", "spisz", min_support=3
        ).features
        effects = [abs(f.log_odds) for f in features]
        assert effects == sorted(effects, reverse=True)

    def test_holm_never_below_raw_p(self) -> None:
        for f in contrast_dialects(_podhale_vs_spisz(), "podhale", "spisz", min_support=3).features:
            assert f.p_value_holm >= f.p_value - 1e-9

    def test_min_support_excludes_rare_features(self) -> None:
        # A one-off rare token appears in a single doc; a high threshold drops it.
        samples = _podhale_vs_spisz()
        samples[0] = Sample(
            id="rare",
            text="hej baca zupelnieunikatoweslowo",
            labels=DialectLabels(dialect="podhale"),
        )
        report = contrast_dialects(samples, "podhale", "spisz", min_support=5)
        assert not any("unikatowe" in f.feature for f in report.features)

    def test_family_level_contrast(self) -> None:
        a = _samples("x", ["baca hej owce", "baca na holi"] * 4, level_family="lesser_polish")
        b = _samples("y", ["dzien dobry prosze", "jak sie masz"] * 4, level_family="standard")
        report = contrast_dialects(
            a + b, "lesser_polish", "standard", level=LabelLevel.FAMILY, min_support=3
        )
        assert report.level == "family"
        assert {"lesser_polish", "standard"} == {report.dialect_a, report.dialect_b}


class TestValidation:
    def test_rejects_self_contrast(self) -> None:
        with pytest.raises(ConfigurationError, match="itself"):
            contrast_dialects([], "podhale", "podhale")

    def test_rejects_absent_dialect(self) -> None:
        with pytest.raises(ConfigurationError, match="no text samples"):
            contrast_dialects(_samples("podhale", ["hej"]), "podhale", "spisz", min_support=1)

    def test_rejects_bad_params(self) -> None:
        with pytest.raises(ConfigurationError, match="min_support"):
            contrast_dialects(_podhale_vs_spisz(), "podhale", "spisz", min_support=0)


class TestRendering:
    def test_markdown_byte_stable_and_sectioned(self) -> None:
        report = contrast_dialects(_podhale_vs_spisz(), "podhale", "spisz", min_support=3)
        markdown = report.to_markdown()
        assert markdown == report.to_markdown()
        assert "# Contrastive analysis: Podhale vs Spisz" in markdown
        assert "## Lexical markers" in markdown
        assert "## Phonological isoglosses" in markdown
        assert "## Morphological endings" in markdown

    def test_save_round_trips(self, tmp_path: Path) -> None:
        report = contrast_dialects(_podhale_vs_spisz(), "podhale", "spisz", min_support=3)
        path = tmp_path / "contrast.json"
        report.save(path)
        reloaded = ContrastReport.model_validate_json(path.read_text(encoding="utf-8"))
        assert reloaded.dialect_a == "podhale" and len(reloaded.features) == len(report.features)
