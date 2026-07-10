"""Tests for tulip.features.text.phonology.

Ground truth for the corpus-level assertions comes from
:func:`tulip.data.synthetic.generate_corpus`, whose Kurpie class applies the
exact ``pi/bi/wi/mi -> psi/bzi/wzi/mni`` (soft-labial) and ``cz/sz/ż/dż ->
c/s/z/dz`` (mazurzenie) transforms this extractor is meant to pick up, so the
sign and magnitude of each feature per class are known in advance.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from tulip.core.exceptions import ConfigurationError
from tulip.data.synthetic import SyntheticSpec, generate_corpus

if TYPE_CHECKING:
    from pathlib import Path


def _import_guard() -> None:
    """Keep tulip.features importable before the sibling audio package exists."""
    try:
        import tulip.features
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on build order
        if exc.name != "tulip.features.audio":
            raise
        sys.modules["tulip.features.audio"] = types.ModuleType("tulip.features.audio")
        import tulip.features  # noqa: F401


_import_guard()

from tulip.config.schemas import ComponentConfig  # noqa: E402
from tulip.features.registries import TEXT_FEATURES  # noqa: E402
from tulip.features.text import (  # noqa: E402
    PhonologicalMarkerExtractor,
    build_text_features,
)

_SOFT = "phon:soft_labial_cluster"
_SIB = "phon:sibilant_digraph"


def _fitted() -> PhonologicalMarkerExtractor:
    """A bundled-isogloss extractor; fit ignores its X (isoglosses define columns)."""
    return PhonologicalMarkerExtractor().fit([])


def _col(extractor: PhonologicalMarkerExtractor, name: str) -> int:
    return list(extractor.get_feature_names_out()).index(name)


def _class_mean(spec: SyntheticSpec, feature: str, *, family: str | None = None) -> float:
    """Mean of ``feature`` over a synthetic corpus, optionally one label family."""
    extractor = _fitted()
    column = _col(extractor, feature)
    corpus = generate_corpus(spec)
    if family is not None:
        corpus = [sample for sample in corpus if sample.labels.family == family]
    matrix = extractor.transform([sample.text or "" for sample in corpus])
    return float(matrix[:, column].mean())


def _write_isoglosses(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "isoglosses.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------- registration


def test_registered_in_text_features() -> None:
    assert "phonological_markers" in TEXT_FEATURES.names()


# ------------------------------------------------------- corpus ground truth


def test_soft_labial_high_for_kurpie_and_zero_elsewhere() -> None:
    """Kurpie respells labials (psi/bzi/wzi/mni); untransformed classes cannot."""
    kurpie = _class_mean(
        SyntheticSpec(dialects=("kurpie",), include_standard=False, marker_dropout=0.0, seed=7),
        _SOFT,
    )
    podhale = _class_mean(
        SyntheticSpec(dialects=("podhale",), include_standard=False, marker_dropout=0.0, seed=7),
        _SOFT,
    )
    standard = _class_mean(
        SyntheticSpec(dialects=("kurpie",), include_standard=True, marker_dropout=0.0, seed=7),
        _SOFT,
        family="standard",
    )
    assert kurpie > 0.05  # a respelt cluster in >5% of tokens
    # Untransformed classes hit zero: their only cluster-bearing word
    # (``ziemniaki`` -> ``mni``) is on the exclude stoplist.
    assert podhale == 0.0
    assert standard == 0.0


def test_sibilant_digraph_stripped_by_mazurzenie_in_kurpie() -> None:
    """Mazurzenie deletes cz/sz/ż/dż, so Kurpie's digraph rate collapses to ~0."""
    kurpie = _class_mean(
        SyntheticSpec(dialects=("kurpie",), include_standard=False, marker_dropout=0.0, seed=7),
        _SIB,
    )
    standard = _class_mean(
        SyntheticSpec(dialects=("kurpie",), include_standard=True, marker_dropout=0.0, seed=7),
        _SIB,
        family="standard",
    )
    assert standard > 0.05  # standard Polish keeps its sibilant digraphs
    assert kurpie < 0.1 * standard  # mazurzenie strips them: conspicuously low
    assert kurpie < 0.01


# --------------------------------------------------------- exclusion / precision


def test_standard_words_are_not_soft_labial_false_positives() -> None:
    """Hand-written standard sentences with mnie/psa/psie/wino yield ZERO hits.

    This is the assertion that proves the exclude stoplist works: every one of
    these tokens contains, or looks like it contains, a targeted cluster, yet a
    naive regex firing on any of them would make the feature pure noise.
    """
    sentences = [
        "Podaj mnie psa, a psie zostaw przy budzie.",
        "Muszę wziąć więcej mniejszych ziemniaków na zimę.",
        "Zostało jeszcze dobre wino w piwnicy.",
    ]
    extractor = _fitted()
    column = _col(extractor, _SOFT)
    matrix = extractor.transform(sentences)
    assert np.all(matrix[:, column] == 0.0)


def test_exclusion_is_selective_not_blanket() -> None:
    """Dialectal respellings score; the standard collisions they resemble do not."""
    extractor = _fitted()
    column = _col(extractor, _SOFT)
    dialectal = extractor.transform(["psiwo kobzieta bziały"])[0, column]
    standard = extractor.transform(["psie mnie wziąć"])[0, column]
    assert dialectal > 0.0
    assert standard == 0.0


def test_soft_labial_rate_is_hand_computable() -> None:
    """One dialectal cluster among four tokens = 1/4 of ``per_tokens``."""
    extractor = _fitted()
    column = _col(extractor, _SOFT)
    assert extractor.transform(["psiwo abc def ghi"])[0, column] == 0.25  # 1 hit / 4 tokens * 1.0


def test_per_tokens_scales_the_rate_linearly() -> None:
    """The default is a per-token fraction; per_tokens is a pure reporting scale."""
    text = ["psiwo abc def ghi"]
    default = PhonologicalMarkerExtractor().fit([])
    reported = PhonologicalMarkerExtractor(per_tokens=1000.0).fit([])
    column = _col(default, _SOFT)
    assert default.transform(text)[0, column] == 0.25
    assert reported.transform(text)[0, column] == 250.0


def test_default_scale_is_comparable_to_tfidf() -> None:
    """Dense phon columns must not dwarf the sparse TF-IDF block they union with.

    A column scaled 1000x larger is effectively 1000x less L2-regularised and
    drowns thousands of TF-IDF columns; measured, it flipped a +0.02 accuracy
    gain into a -0.01 loss. Pin the default so that regression cannot return.
    """
    corpus = generate_corpus(SyntheticSpec(n_speakers_per_dialect=2, samples_per_speaker=3, seed=7))
    rows = PhonologicalMarkerExtractor().fit([]).transform([s.text for s in corpus])
    assert rows.max() <= 1.0


# ----------------------------------------------------------- sklearn contract


def test_fit_transform_shape_and_feature_names() -> None:
    texts = ["psiwo kobzieta", "zwykly tekst po polsku", ""]
    extractor = PhonologicalMarkerExtractor()
    matrix = extractor.fit(texts).transform(texts)
    names = extractor.get_feature_names_out()
    assert list(names) == [_SOFT, _SIB]
    assert matrix.shape == (len(texts), len(names))
    assert np.all(np.isfinite(matrix))


def test_transform_before_fit_raises_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        PhonologicalMarkerExtractor().transform(["psiwo"])


def test_feature_names_before_fit_raises_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        PhonologicalMarkerExtractor().get_feature_names_out()


def test_works_inside_build_text_features() -> None:
    union = build_text_features([ComponentConfig(name="phonological_markers")])
    matrix = union.fit_transform(["psiwo kobzieta", "zwykly tekst po polsku"])
    assert matrix.shape == (2, 2)
    names = list(union.get_feature_names_out())
    assert any(name.endswith(_SOFT) for name in names)
    assert any(name.endswith(_SIB) for name in names)


# ---------------------------------------------------------- degenerate inputs


def test_empty_and_whitespace_documents_are_zero_rows() -> None:
    extractor = _fitted()
    matrix = extractor.transform(["", "   ", "\n\t"])
    assert matrix.shape == (3, 2)
    assert np.all(matrix == 0.0)
    assert np.all(np.isfinite(matrix))


def test_empty_input_sequence_keeps_column_count() -> None:
    matrix = _fitted().transform([])
    assert matrix.shape == (0, 2)


# ------------------------------------------------- custom isogloss files (OCP)


def test_custom_digraph_isogloss_counts_per_tokens(tmp_path: Path) -> None:
    path = _write_isoglosses(
        tmp_path,
        "version: 1\nisoglosses:\n  - name: cz_only\n    kind: digraph\n    digraphs: [cz]\n",
    )
    extractor = PhonologicalMarkerExtractor(isogloss_path=path, per_tokens=100.0).fit([])
    assert list(extractor.get_feature_names_out()) == ["phon:cz_only"]
    # "czapka czas" -> 2 tokens, 2 'cz' occurrences -> 2/2 * 100.
    assert extractor.transform(["czapka czas"])[0, 0] == 100.0


def test_longest_digraph_wins_no_double_count() -> None:
    """``dż`` must be consumed whole, not counted as ``dż`` plus a nested ``ż``."""
    extractor = _fitted()
    column = _col(extractor, _SIB)
    # "dżem" is one token with one digraph occurrence -> 1/1 * 1.0.
    assert extractor.transform(["dżem"])[0, column] == 1.0


# ------------------------------------------------------ configuration errors


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        PhonologicalMarkerExtractor(isogloss_path=tmp_path / "nope.yaml").fit([])


def test_unknown_kind_raises(tmp_path: Path) -> None:
    path = _write_isoglosses(
        tmp_path,
        "version: 1\nisoglosses:\n  - name: x\n    kind: bogus\n    digraphs: [cz]\n",
    )
    with pytest.raises(ConfigurationError, match="unknown kind"):
        PhonologicalMarkerExtractor(isogloss_path=path).fit([])


def test_invalid_regex_raises(tmp_path: Path) -> None:
    path = _write_isoglosses(
        tmp_path,
        'version: 1\nisoglosses:\n  - name: x\n    kind: pattern\n    pattern: "(unclosed"\n',
    )
    with pytest.raises(ConfigurationError, match="invalid"):
        PhonologicalMarkerExtractor(isogloss_path=path).fit([])


def test_non_positive_per_tokens_raises() -> None:
    with pytest.raises(ConfigurationError, match="per_tokens"):
        PhonologicalMarkerExtractor(per_tokens=0.0).fit([])


def test_empty_isogloss_list_raises(tmp_path: Path) -> None:
    path = _write_isoglosses(tmp_path, "version: 1\nisoglosses: []\n")
    with pytest.raises(ConfigurationError, match="non-empty list"):
        PhonologicalMarkerExtractor(isogloss_path=path).fit([])


def test_unsupported_schema_version_raises(tmp_path: Path) -> None:
    path = _write_isoglosses(
        tmp_path,
        "version: 99\nisoglosses:\n  - name: x\n    kind: digraph\n    digraphs: [cz]\n",
    )
    with pytest.raises(ConfigurationError, match="version"):
        PhonologicalMarkerExtractor(isogloss_path=path).fit([])


def test_duplicate_isogloss_name_raises(tmp_path: Path) -> None:
    path = _write_isoglosses(
        tmp_path,
        "version: 1\nisoglosses:\n"
        "  - name: dup\n    kind: digraph\n    digraphs: [cz]\n"
        "  - name: dup\n    kind: digraph\n    digraphs: [sz]\n",
    )
    with pytest.raises(ConfigurationError, match="duplicate"):
        PhonologicalMarkerExtractor(isogloss_path=path).fit([])


def test_empty_digraph_list_raises(tmp_path: Path) -> None:
    path = _write_isoglosses(
        tmp_path,
        "version: 1\nisoglosses:\n  - name: x\n    kind: digraph\n    digraphs: []\n",
    )
    with pytest.raises(ConfigurationError, match="digraphs"):
        PhonologicalMarkerExtractor(isogloss_path=path).fit([])
