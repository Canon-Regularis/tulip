"""Tests for the phonological rule engine (apply/normalize + applicable/fired)."""

from __future__ import annotations

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError
from tulip.features.text.phonological_rules import (
    PhonologicalRuleExtractor,
    apply_rules,
    load_phonological_rules,
    normalize_to_standard,
)


@pytest.fixture(scope="module")
def rules():
    return load_phonological_rules()


def _rule(rules, name):
    return next(r for r in rules if r.name == name)


class TestBundledRuleSet:
    def test_expected_rules_load(self, rules) -> None:
        names = {r.name for r in rules}
        assert {"mazurzenie", "soft_labials", "silesian_final_ch", "kaszubienie"} <= names

    def test_merger_is_not_detectable_and_soft_labials_is(self, rules) -> None:
        assert _rule(rules, "mazurzenie").detectable is False
        assert _rule(rules, "kaszubienie").detectable is False
        assert _rule(rules, "soft_labials").detectable is True


class TestForwardApplication:
    def test_mazurzenie_forward(self, rules) -> None:
        maz = _rule(rules, "mazurzenie")
        # jeszcze -> jesce, czego -> cego, dżem -> dzem (dż before ż).
        assert maz.apply_token("jeszcze") == "jesce"
        assert maz.apply_token("czego") == "cego"
        assert maz.apply_token("dżem") == "dzem"

    def test_soft_labials_forward_and_stoplist(self, rules) -> None:
        soft = _rule(rules, "soft_labials")
        assert soft.apply_token("piwo") == "psiwo"
        assert soft.apply_token("kobieta") == "kobzieta"
        # A standard collision on the stoplist is never rewritten.
        assert soft.apply_token("mnie") == "mnie"
        assert soft.apply_token("ziemniak") == "ziemniak"

    def test_final_position_only(self, rules) -> None:
        sil = _rule(rules, "silesian_final_ch")
        assert sil.apply_token("dach") == "dak"  # final -ch
        assert sil.apply_token("chleb") == "chleb"  # initial ch untouched


class TestReverseNormalisation:
    def test_detectable_rule_reverses(self, rules) -> None:
        # psiwo -> piwo, kobzieta -> kobieta (soft labials are detectable).
        assert normalize_to_standard("psiwo kobzieta", rules=rules) == "piwo kobieta"

    def test_merger_is_left_untouched(self, rules) -> None:
        # mazurzenie is lossy: 'jesce' cannot be restored to 'jeszcze'.
        assert normalize_to_standard("jesce", rules=rules) == "jesce"

    def test_normalisation_collapses_a_dialectal_variant(self, rules) -> None:
        # The dialectal and standard spellings normalise to the same string.
        assert normalize_to_standard("psiwo", rules=rules) == normalize_to_standard(
            "piwo", rules=rules
        )


class TestApplicableVsFired:
    def test_soft_labials_distinguishes_environment_from_change(self, rules) -> None:
        soft = _rule(rules, "soft_labials")
        standard = ["piwo", "biały"]  # environment present, change absent
        dialectal = ["psiwo", "bziały"]  # change fired
        assert soft.applicable_rate(standard) > 0 and soft.fired_rate(standard) == 0
        assert soft.fired_rate(dialectal) > 0 and soft.applicable_rate(dialectal) == 0

    def test_merger_reports_only_applicable(self, rules) -> None:
        maz = _rule(rules, "mazurzenie")
        assert maz.applicable_rate(["jeszcze"]) > 0
        assert maz.fired_rate(["jeszcze"]) == 0.0  # mergers never expose 'fired'
        assert maz.fired_rate(["jesce"]) == 0.0


class TestExtractor:
    def test_columns_cover_applicable_and_fired(self) -> None:
        extractor = PhonologicalRuleExtractor().fit(["placeholder"])
        names = list(extractor.get_feature_names_out())
        assert "rule:mazurzenie:applicable" in names
        assert "rule:mazurzenie:fired" not in names  # merger: applicable only
        assert "rule:soft_labials:applicable" in names
        assert "rule:soft_labials:fired" in names

    def test_transform_shape_and_signal(self) -> None:
        extractor = PhonologicalRuleExtractor().fit(["placeholder"])
        matrix = extractor.transform(["psiwo psiwo", ""])
        assert matrix.shape == (2, len(extractor.feature_names_))
        fired_col = list(extractor.feature_names_).index("rule:soft_labials:fired")
        assert matrix[0, fired_col] > 0  # soft labials fired
        assert np.all(matrix[1] == 0)  # empty doc -> all zeros

    def test_rejects_non_positive_per_tokens(self) -> None:
        with pytest.raises(ConfigurationError, match="per_tokens"):
            PhonologicalRuleExtractor(per_tokens=0).fit(["x"])


class TestGeneratorConsistency:
    """DRY safeguard: the engine's forward rules must agree with the corpus
    generator's transforms and with the isoglosses.yaml detectors, so the same
    linguistic fact cannot silently diverge across its representations."""

    def test_forward_matches_generator_transforms(self, rules) -> None:
        from tulip.data._synthetic_corpus import MAZURZENIE, SOFT_LABIALS

        maz = _rule(rules, "mazurzenie")
        soft = _rule(rules, "soft_labials")
        # Every lowercase (standard -> dialectal) pair the generator applies must
        # be reproduced by the engine's forward rewrite.
        for standard, dialectal in MAZURZENIE:
            if standard.islower():
                assert maz.apply_token(standard) == dialectal
        for standard, dialectal in SOFT_LABIALS:
            if standard.islower():
                # The bare cluster (e.g. 'pi') rewrites to its reflex ('psi').
                assert soft.apply_token(standard) == dialectal

    def test_mazurzenie_digraphs_match_isoglosses(self, rules) -> None:
        # The engine's mazurzenie source graphemes are exactly the standard
        # sibilant digraphs isoglosses.yaml measures the rate of.
        maz = _rule(rules, "mazurzenie")
        engine_sources = {standard for standard, _ in maz.pairs}
        assert engine_sources == {"cz", "sz", "ż", "dż"}


def test_apply_rules_module_helper() -> None:
    # The module-level forward helper round-trips through the bundled set.
    assert "psiwo" in apply_rules("piwo")
