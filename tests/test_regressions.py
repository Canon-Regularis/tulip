"""Regression guards for the defects the codebase bug hunts turned up.

Each test is named for the failure it stops from coming back, one per fixed bug,
spanning the data, evaluation, models, serve, and pipeline layers. A fix that its
own subsystem suite already exercises is not repeated here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tulip.core.exceptions import ConfigurationError, DataError


def test_label_encoded_classifier_is_a_real_sklearn_estimator() -> None:
    # Wrapping an integer-label estimator must not lose the sklearn tags: without
    # them, Pipeline.predict and is_classifier crash on sklearn >= 1.6, disabling
    # the xgboost/lightgbm baselines and their use as ensemble members.
    from sklearn.base import is_classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from tulip.models.classical import LabelEncodedClassifier

    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["a", "a", "b", "b"])
    wrapped = LabelEncodedClassifier(LogisticRegression())
    assert is_classifier(wrapped)
    pipeline = Pipeline([("model", wrapped)]).fit(x, y)
    assert list(pipeline.predict(x)) == ["a", "a", "b", "b"]
    assert pipeline.predict_proba(x).shape == (4, 2)


def test_malformed_lexicon_yaml_raises_configuration_error(tmp_path) -> None:
    from tulip.features.text.keywords import load_lexicon

    bad = tmp_path / "bad.yaml"
    bad.write_text("podhale: [baca, kaj\nsilesia: broken", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_lexicon(bad)


def test_lexicon_duplicate_key_after_casefold_raises(tmp_path) -> None:
    from tulip.features.text.keywords import load_lexicon

    dup = tmp_path / "dup.yaml"
    dup.write_text("Podhale: [baca]\npodhale: [kaj]\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate"):
        load_lexicon(dup)


def test_end_of_word_bpe_pieces_merge_into_words() -> None:
    from tulip.explain.attention import _merge_by_markers

    tokens = ["low", "er</w>", "dog</w>"]
    weights = np.array([0.5, 0.3, 0.2])
    merged = _merge_by_markers(tokens, weights, special_tokens=set())
    assert [word for word, _ in merged] == ["lower", "dog"]
    assert merged[0][1] == pytest.approx(0.8)


def test_split_fingerprint_load_reports_dataerror_on_a_bad_lock(tmp_path) -> None:
    from tulip.data.fingerprint import SplitFingerprint

    corrupt = tmp_path / "lock.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DataError):
        SplitFingerprint.load(corrupt)

    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text('{"combined": "abc", "digests": "not a mapping"}', encoding="utf-8")
    with pytest.raises(DataError):
        SplitFingerprint.load(wrong_shape)


def test_asr_multichar_variant_is_title_cased_not_upper() -> None:
    from tulip.robustness.perturbations import _substitute

    # A capitalised source with a multi-character variant becomes "Rz", not "RZ".
    assert _substitute("Z", 1.0, np.random.default_rng(0), {"z": ("rz",)}) == "Rz"
    assert _substitute("z", 1.0, np.random.default_rng(0), {"z": ("rz",)}) == "rz"


def test_project_embeddings_rejects_ragged_input_with_dataerror() -> None:
    from tulip.viz.embedding_space import _as_dense_2d

    with pytest.raises(DataError):
        _as_dense_2d([[1.0, 2.0], [3.0]])


def test_cvconfig_rejects_empty_seeds() -> None:
    from pydantic import ValidationError

    from tulip.pipeline import CVConfig

    with pytest.raises(ValidationError):
        CVConfig(seeds=())


def test_non_utf8_manifest_reports_a_clean_error(tmp_path) -> None:
    from tulip.data.manifest import ManifestColumns, read_manifest

    # A Polish CSV saved as CP1250 (the Excel norm): 'ł' becomes byte 0xB3, an
    # invalid UTF-8 start byte. The loader must not die with a raw UnicodeDecodeError.
    bad = tmp_path / "manifest.csv"
    bad.write_bytes("id,text\n1,godał\n".encode("cp1250"))
    with pytest.raises(DataError):
        list(read_manifest(bad, columns=ManifestColumns()))


def test_serve_settings_reject_a_bad_env_var() -> None:
    from tulip.serve.settings import ServeSettings

    with pytest.raises(ConfigurationError):
        ServeSettings.from_env({"TULIP_SERVE_RATE_LIMIT": "not-a-number"})
    with pytest.raises(ConfigurationError):
        ServeSettings.from_env({"TULIP_SERVE_MAX_BATCH": "100000"})  # over the ceiling


def test_split_predictions_load_reports_configuration_error(tmp_path) -> None:
    from tulip.evaluation.predictions import SplitPredictions

    with pytest.raises(ConfigurationError):
        SplitPredictions.load(tmp_path / "missing.json")
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        SplitPredictions.load(corrupt)


def test_split_fingerprint_load_missing_file_reports_dataerror(tmp_path) -> None:
    from tulip.data.fingerprint import SplitFingerprint

    with pytest.raises(DataError):
        SplitFingerprint.load(tmp_path / "does-not-exist.json")


def test_audio_sample_digest_is_path_portable() -> None:
    from tulip.core.types import DialectLabels, Sample
    from tulip.data.fingerprint import sample_digest

    labels = DialectLabels(dialect="podhale")
    windows = Sample(id="s1", audio_path=r"C:\corpus\audio\clip.wav", labels=labels)
    posix = Sample(id="s1", audio_path="/srv/corpus/audio/clip.wav", labels=labels)
    assert sample_digest(windows) == sample_digest(posix)


def test_calibration_tolerates_a_probability_at_the_tolerance_edge() -> None:
    from tulip.evaluation.calibration import compute_calibration

    # A top probability just past 1.0, within the accepted tolerance, must not push
    # a bin's confidence past its bound and raise a non-TulipError.
    proba = np.array([[1.0 + 5e-10, -5e-10], [0.4, 0.6]])
    report = compute_calibration(["a", "b"], proba, ["a", "b"], n_bins=10)
    assert all(0.0 <= b.confidence <= 1.0 for b in report.bins)


def test_metrics_calibration_disables_gracefully_on_a_nan_probability() -> None:
    from tulip.evaluation.metrics import compute_metrics

    proba = np.array([[np.nan, np.nan], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
    report = compute_metrics(
        ["a", "b", "a", "b"], ["a", "b", "b", "b"], proba, labels=["a", "b"], calibration_bins=10
    )
    assert report.calibration is None  # disabled, not a hard failure
    assert report.accuracy == pytest.approx(0.75)


def test_uncertainty_single_member_reports_configuration_error() -> None:
    from tulip.evaluation.uncertainty import decompose_uncertainty

    with pytest.raises(ConfigurationError):
        decompose_uncertainty(np.zeros((1, 3, 2)))  # one member, needs >= 2


def test_generic_manifest_loader_rejects_a_bad_columns_mapping() -> None:
    from tulip.data.registry import DATASETS

    with pytest.raises(DataError):
        DATASETS.create("manifest", columns={"speaker": "who"})  # not a ManifestColumns field


def test_check_per_tokens_rejects_non_finite() -> None:
    from tulip.features.text._base import check_per_tokens

    for bad in (float("nan"), float("inf")):
        with pytest.raises(ConfigurationError):
            check_per_tokens(bad)


def test_age_band_passes_a_non_finite_age_through() -> None:
    from tulip.evaluation.slicing import age_band

    assert age_band("1e999") == "1e999"  # int(float("1e999")) is int(inf) -> OverflowError


def test_load_benchmark_reports_configuration_error(tmp_path) -> None:
    from tulip.evaluation.benchmark import load_benchmark

    with pytest.raises(ConfigurationError):
        load_benchmark(tmp_path / "missing.json")
    corrupt = tmp_path / "benchmark.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_benchmark(corrupt)


def test_project_embeddings_rejects_a_zero_feature_matrix() -> None:
    from tulip.viz.embedding_space import project_embeddings

    with pytest.raises(DataError):
        project_embeddings([[], []], ["a", "b"])


def test_load_datasheet_spec_reports_configuration_error(tmp_path) -> None:
    from tulip.evaluation.datasheet import load_datasheet_spec

    bad = tmp_path / "spec.yaml"
    bad.write_text("motivation:\n  - not\n  - a string\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_datasheet_spec(bad)


def test_read_samples_reports_dataerror_on_corrupt_jsonl(tmp_path) -> None:
    from tulip.data.reading import read_samples

    corrupt = tmp_path / "split.jsonl"
    corrupt.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(DataError):
        list(read_samples(corrupt))


def test_non_utf8_lexicon_reports_configuration_error(tmp_path) -> None:
    from tulip.features.text.keywords import load_lexicon

    bad = tmp_path / "lexicon.yaml"
    bad.write_bytes("podhale: [godał]\n".encode("cp1250"))  # 'ł' -> 0xB3, invalid UTF-8
    with pytest.raises(ConfigurationError):
        load_lexicon(bad)


def test_mcnemar_survives_a_large_discordant_count() -> None:
    from tulip.evaluation.significance import _mcnemar_p, mcnemar_exact

    # The exact binomial sum outgrows the float range past a few hundred discordant
    # pairs; scaling it by 0.5**n forced an int->float conversion that raised
    # OverflowError on any split big enough to matter.
    assert _mcnemar_p(520, 520) == pytest.approx(1.0)
    assert 0.0 < _mcnemar_p(5000, 4000) < 1e-20
    a = [True] * 900 + [False] * 900
    b = [False] * 900 + [True] * 900
    _, _, p = mcnemar_exact(a, b)
    assert p == pytest.approx(1.0)


def test_minimum_detectable_effect_survives_a_large_split() -> None:
    from tulip.evaluation.power import minimum_detectable_effect

    # math.comb(n, i) exceeds the float range around n=1030, so the direct
    # comb * p**i product raised OverflowError on realistic benchmark splits.
    small = minimum_detectable_effect(500).mde
    large = minimum_detectable_effect(2000).mde
    assert small is not None and large is not None
    assert 0.0 < large < small  # more samples detect a smaller effect


def test_load_benchmark_rejects_a_non_list_results_field(tmp_path) -> None:
    import json

    from tulip.evaluation.benchmark import load_benchmark

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"schema_version": 1, "results": 5}), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_benchmark(path)


def test_ensemble_models_without_estimators_report_configuration_error() -> None:
    from tulip.models.registry import MODELS

    for name in ("voting", "stacking"):
        with pytest.raises(ConfigurationError, match="estimators"):
            MODELS.create(name)


def test_dataset_entry_with_an_unknown_param_reports_dataerror() -> None:
    from tulip.config.schemas import ComponentConfig, DataConfig
    from tulip.data.builder import DatasetBuilder

    config = DataConfig(
        root="data/raw", datasets=[ComponentConfig(name="synthetic", params={"nope": 1})]
    )
    with pytest.raises(DataError):
        DatasetBuilder(config).load_samples()


def test_unreadable_manifest_reports_dataerror(tmp_path) -> None:
    from tulip.data.manifest import _iter_csv_rows, _iter_jsonl_rows

    # is_file() upstream cannot vouch for openability (a locked or unreadable file);
    # a directory stands in for the OSError that open() raises in that case.
    with pytest.raises(DataError):
        list(_iter_csv_rows(tmp_path, None))
    with pytest.raises(DataError):
        list(_iter_jsonl_rows(tmp_path))


def test_writers_reject_an_output_root_that_is_a_file(tmp_path) -> None:
    from tulip.data.synthetic import SyntheticSpec, write_synthetic_manifest
    from tulip.data.synthetic_audio import AudioSyntheticSpec, write_synthetic_audio_manifest
    from tulip.data.transcribe import TranscriptCache, write_transcribed_manifest

    occupied = tmp_path / "already_a_file"
    occupied.write_text("x", encoding="utf-8")
    with pytest.raises(DataError):
        write_synthetic_manifest(SyntheticSpec(), occupied)
    with pytest.raises(DataError):
        write_synthetic_audio_manifest(AudioSyntheticSpec(), occupied)
    with pytest.raises(DataError):
        write_transcribed_manifest([], occupied)
    with pytest.raises(DataError):
        TranscriptCache(occupied)


def test_feature_config_with_an_unknown_param_reports_configuration_error() -> None:
    from tulip.features.text.composite import build_text_features

    with pytest.raises(ConfigurationError):
        build_text_features([{"name": "char_tfidf", "params": {"bogus_kw": 1}}])


def test_llm_cache_treats_a_non_object_json_file_as_a_miss(tmp_path) -> None:
    from tulip.models._llm_cache import LLMResponseCache

    cache = LLMResponseCache(tmp_path)
    entry = tmp_path / "k.json"
    for payload in ("null", "[]", '"a string"', "5"):
        entry.write_text(payload, encoding="utf-8")  # valid JSON, not an object
        cache._memory.clear()
        assert cache.get("k") is None  # a miss, not a TypeError
    entry.write_text('{"response": "ok"}', encoding="utf-8")
    cache._memory.clear()
    assert cache.get("k") == "ok"  # a real object still hits


def test_robustness_cli_rejects_a_negative_seed() -> None:
    from typer.testing import CliRunner

    from tulip.cli.app import app

    result = CliRunner().invoke(app, ["robustness", "missing.yaml", "--seed=-1"])
    assert result.exit_code != 0
    assert "seed" in result.output.lower()


def test_marker_density_guards_the_denominator_not_the_marker_set() -> None:
    from tulip.features.text.dialect_intensity import DialectIntensityExtractor

    # The per-token density divides by len(tokens), so the empty-guard must key on
    # tokens. A non-empty marker set with zero tokens must return 0.0, not raise
    # ZeroDivisionError (which the earlier `if not markers` guard let through).
    assert DialectIntensityExtractor._marker_density([], frozenset({"baca"})) == 0.0
    assert DialectIntensityExtractor._marker_density(["baca", "kaj"], frozenset()) == 0.0
