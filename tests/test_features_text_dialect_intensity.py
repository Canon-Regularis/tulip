"""Tests for the dialect-intensity feature (bounded, anchored, composed)."""

from __future__ import annotations

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.features.text.dialect_intensity import DialectIntensityExtractor


@pytest.fixture(scope="module")
def extractor() -> DialectIntensityExtractor:
    return DialectIntensityExtractor().fit(["placeholder"])


def _overall(extractor: DialectIntensityExtractor, text: str) -> float:
    return float(extractor.transform([text])[0, 0])


class TestColumns:
    def test_overall_and_per_family_columns(self, extractor: DialectIntensityExtractor) -> None:
        names = list(extractor.get_feature_names_out())
        assert names[0] == "intensity:overall"
        # Families derive from the taxonomy, so the Masovian family is present.
        assert "intensity:masovian" in names
        assert all(name.startswith("intensity:") for name in names)


class TestAnchorAndBounds:
    def test_standard_text_scores_zero(self, extractor: DialectIntensityExtractor) -> None:
        # Pure standard Polish: no marker, no fired sound change -> exactly 0.
        assert _overall(extractor, "wczoraj poszedł do sklepu po chleb") == pytest.approx(0.0)

    def test_empty_text_is_all_zero(self, extractor: DialectIntensityExtractor) -> None:
        assert np.all(extractor.transform([""]) == 0.0)

    def test_scores_stay_in_unit_interval(self, extractor: DialectIntensityExtractor) -> None:
        matrix = extractor.transform(["baca baca psiwo psiwo gryfny kaj", "standardowy tekst"])
        assert np.all(matrix >= 0.0) and np.all(matrix < 1.0)


class TestSignal:
    def test_markers_raise_intensity(self, extractor: DialectIntensityExtractor) -> None:
        # 'baca' is a Podhale marker; a text carrying it must beat standard text.
        assert _overall(extractor, "hej baca poszedł na hale") > _overall(
            extractor, "hej poszedł na pole"
        )

    def test_fired_sound_change_raises_intensity(
        self, extractor: DialectIntensityExtractor
    ) -> None:
        # 'psiwo' fires the (detectable) soft-labial rule; 'piwo' does not.
        assert _overall(extractor, "przyniósł psiwo") > _overall(extractor, "przyniósł piwo")

    def test_more_evidence_scores_higher(self, extractor: DialectIntensityExtractor) -> None:
        one = _overall(extractor, "baca poszedł")
        two = _overall(extractor, "baca gryfny poszedł")
        assert two > one

    def test_masovian_family_responds_to_soft_labials(
        self, extractor: DialectIntensityExtractor
    ) -> None:
        names = list(extractor.feature_names_)
        masovian = names.index("intensity:masovian")
        row = extractor.transform(["kobzieta warzy psiwo"])[0]
        assert row[masovian] > 0.0


class TestValidation:
    def test_rejects_negative_weight(self) -> None:
        with pytest.raises(ConfigurationError, match="weight"):
            DialectIntensityExtractor(marker_weight=-1.0).fit(["x"])

    def test_determinism(self) -> None:
        a = DialectIntensityExtractor().fit(["x"]).transform(["baca psiwo"])
        b = DialectIntensityExtractor().fit(["x"]).transform(["baca psiwo"])
        np.testing.assert_array_equal(a, b)
