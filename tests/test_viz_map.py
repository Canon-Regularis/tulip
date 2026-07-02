"""Tests for tulip.viz.map (all require folium, extra ``viz``)."""

from __future__ import annotations

import logging

import pytest

folium = pytest.importorskip("folium")

from tulip.core.types import ClassProbability, Prediction  # noqa: E402
from tulip.labels.taxonomy import LabelLevel, RegionalDialect, display_name  # noqa: E402
from tulip.viz.map import confidence_heatmap, prediction_map, save_map  # noqa: E402


def _region_prediction() -> Prediction:
    return Prediction(
        label="podhale",
        level=LabelLevel.DIALECT,
        probabilities=(
            ClassProbability(label="podhale", probability=0.61),
            ClassProbability(label="silesia", probability=0.25),
            ClassProbability(label="kurpie", probability=0.14),
        ),
    )


def _rendered(fmap: folium.Map) -> str:
    return fmap.get_root().render()


def test_prediction_map_contains_names_and_percentages() -> None:
    fmap = prediction_map(_region_prediction())
    assert isinstance(fmap, folium.Map)
    html = _rendered(fmap)
    assert "Podhale" in html
    assert "61.0%" in html
    assert "Silesia" in html
    assert "25.0%" in html
    assert "Kurpie" in html
    assert "14.0%" in html
    # Family appears in the rich tooltip.
    assert "Lesser Polish" in html


def test_prediction_map_polish_labels() -> None:
    html = _rendered(prediction_map(_region_prediction(), polish_labels=True))
    assert "gwara podhalanska" in html


def test_prediction_map_respects_top_k() -> None:
    html = _rendered(prediction_map(_region_prediction(), top_k=1))
    assert "Podhale" in html
    assert "Silesia" not in html
    assert "Kurpie" not in html


def test_prediction_map_unknown_region_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prediction = Prediction(
        label="atlantis",
        level=LabelLevel.REGION,
        probabilities=(
            ClassProbability(label="atlantis", probability=0.9),
            ClassProbability(label="podhale", probability=0.1),
        ),
    )
    with caplog.at_level(logging.WARNING, logger="tulip"):
        fmap = prediction_map(prediction)
    assert isinstance(fmap, folium.Map)
    assert "Podhale" in _rendered(fmap)
    assert any("atlantis" in record.message for record in caplog.records)


def test_prediction_map_all_unknown_returns_base_map() -> None:
    prediction = Prediction(
        label="nowhere",
        probabilities=(ClassProbability(label="nowhere", probability=1.0),),
    )
    assert isinstance(prediction_map(prediction), folium.Map)


def test_prediction_map_abstained_shows_note_and_faint_regions() -> None:
    prediction = Prediction(
        label=None,
        abstained=True,
        probabilities=(
            ClassProbability(label="podhale", probability=0.2),
            ClassProbability(label="silesia", probability=0.18),
        ),
    )
    html = _rendered(prediction_map(prediction))
    assert "abstained" in html
    # Every catalogued region is drawn faintly, not just the predicted ones.
    for region in RegionalDialect:
        assert display_name(region) in html


def test_prediction_map_voivodeship_level() -> None:
    prediction = Prediction(
        label="mazowieckie",
        level=LabelLevel.VOIVODESHIP,
        probabilities=(
            ClassProbability(label="mazowieckie", probability=0.7),
            ClassProbability(label="malopolskie", probability=0.3),
        ),
    )
    html = _rendered(prediction_map(prediction))
    assert "Mazowieckie" in html
    assert "70.0%" in html
    assert "Malopolskie" in html


def test_confidence_heatmap_includes_all_catalogued_regions() -> None:
    fmap = confidence_heatmap(_region_prediction())
    assert isinstance(fmap, folium.Map)
    html = _rendered(fmap)
    for region in RegionalDialect:
        assert display_name(region) in html
    assert "61.0%" in html


def test_confidence_heatmap_voivodeship_level() -> None:
    prediction = Prediction(
        label="slaskie",
        level=LabelLevel.VOIVODESHIP,
        probabilities=(ClassProbability(label="slaskie", probability=0.8),),
    )
    html = _rendered(confidence_heatmap(prediction))
    assert "Slaskie" in html
    assert "80.0%" in html
    assert "Zachodniopomorskie" in html  # zero-probability regions are still shaded


def test_confidence_heatmap_unknown_label_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prediction = Prediction(
        label="atlantis",
        probabilities=(ClassProbability(label="atlantis", probability=1.0),),
    )
    with caplog.at_level(logging.WARNING, logger="tulip"):
        fmap = confidence_heatmap(prediction)
    assert isinstance(fmap, folium.Map)
    assert any("atlantis" in record.message for record in caplog.records)


def test_save_map_writes_html(tmp_path) -> None:
    fmap = prediction_map(_region_prediction())
    target = tmp_path / "maps" / "prediction.html"
    result = save_map(fmap, target)
    assert result == target
    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "Podhale" in content
