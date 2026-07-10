"""Folium map builders for dialect predictions.

Two views of one :class:`~tulip.core.types.Prediction`:

- :func:`prediction_map` highlights the top-k predicted regions with graded
  circle markers (the winner strongest);
- :func:`confidence_heatmap` shades every region with known geometry by its
  predicted probability.

Both auto-detect the geographic lookup table from ``prediction.level``:
voivodeship-level predictions use :data:`~tulip.labels.geo.VOIVODESHIP_CENTROIDS`,
everything else uses :data:`~tulip.labels.geo.REGION_CENTROIDS`. Labels without
known geometry (corpus-specific regions, family-level labels) are skipped with
a logged warning, never an exception, so a map is always produced.

folium is optional (extra ``viz``) and imported lazily; importing this module
never requires it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from tulip.core.exceptions import ConfigurationError
from tulip.labels.geo import (
    POLAND_BOUNDS,
    POLAND_CENTER,
    REGION_CENTROIDS,
    VOIVODESHIP_CENTROIDS,
    GeoPoint,
    region_centroid,
    voivodeship_centroid,
)
from tulip.labels.taxonomy import LabelLevel, display_name, family_for
from tulip.utils.io import ensure_dir
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    import folium

    from tulip.core.types import Prediction

logger = get_logger(__name__)

#: Single-hue sequential ramp (light -> dark blue) for probability shading.
#: One hue whose lightness carries magnitude; the lightest step recedes toward
#: the basemap so near-zero regions stay quiet.
_SEQUENTIAL_BLUES: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

#: Rank-graded colours for the top-k markers: the winner gets the darkest step,
#: runners-up progressively lighter steps of the same hue. Ranks beyond the
#: table clamp to the lightest entry.
_RANK_COLORS: tuple[str, ...] = ("#104281", "#2a78d6", "#6da7ec", "#9ec5f4", "#cde2fb")

#: Faint styling for abstained predictions (everything recedes; nothing "wins").
_ABSTAIN_COLOR = "#b7d3f6"

_TILES = "cartodbpositron"


def _base_map(folium_mod: object) -> folium.Map:
    """Create a map centred on Poland and fitted to its bounding box."""
    fmap = folium_mod.Map(  # type: ignore[attr-defined]
        location=[POLAND_CENTER.lat, POLAND_CENTER.lon],
        tiles=_TILES,
        zoom_start=6,
        control_scale=True,
    )
    south, west, north, east = POLAND_BOUNDS
    fmap.fit_bounds([[south, west], [north, east]])
    return fmap


def _centroid_for(label: str, level: LabelLevel) -> GeoPoint | None:
    """Look up a label's centroid in the table matching the prediction level."""
    if level is LabelLevel.VOIVODESHIP:
        return voivodeship_centroid(label)
    return region_centroid(label)


def _centroid_table(level: LabelLevel) -> dict[str, GeoPoint]:
    """Return every catalogued ``label -> centroid`` pair for a prediction level."""
    if level is LabelLevel.VOIVODESHIP:
        return dict(VOIVODESHIP_CENTROIDS)
    return {region.value: point for region, point in REGION_CENTROIDS.items()}


def _tooltip_html(label: str, probability: float | None, *, polish: bool) -> str:
    """Build the rich hover tooltip: display name, percent probability, family."""
    lines = [f"<b>{display_name(label, polish=polish)}</b>"]
    if probability is not None:
        lines.append(f"Probability: {probability:.1%}")
    family = family_for(label)
    if family is not None:
        lines.append(f"Family: {display_name(family, polish=polish)}")
    body = "<br>".join(lines)
    return f'<div style="font-family: system-ui, sans-serif; font-size: 13px;">{body}</div>'


def _shade(probability: float) -> tuple[str, float]:
    """Map a probability to a ramp colour and fill opacity.

    Perceived intensity grows roughly with the square root of physical
    intensity (Stevens' power law, exponent ~0.5 for brightness/area), so the
    probability is passed through ``sqrt`` before indexing the ramp and setting
    opacity. This spreads the visually useful range across low probabilities --
    where a many-class softmax lives -- instead of leaving everything but the
    winner indistinguishably faint.
    """
    perceptual = math.sqrt(max(0.0, min(1.0, probability)))
    color = _SEQUENTIAL_BLUES[round(perceptual * (len(_SEQUENTIAL_BLUES) - 1))]
    fill_opacity = 0.15 + 0.65 * perceptual
    return color, fill_opacity


def _add_abstention_layer(
    folium_mod: object, fmap: folium.Map, prediction: Prediction, *, polish: bool
) -> None:
    """Render every known region faintly and attach an abstention note."""
    probabilities = {label.strip().lower(): p for label, p in prediction.as_dict().items()}
    for label, centroid in _centroid_table(prediction.level).items():
        folium_mod.CircleMarker(  # type: ignore[attr-defined]
            location=[centroid.lat, centroid.lon],
            radius=8,
            color=_ABSTAIN_COLOR,
            weight=1,
            fill=True,
            fill_color=_ABSTAIN_COLOR,
            fill_opacity=0.2,
            opacity=0.5,
            tooltip=folium_mod.Tooltip(  # type: ignore[attr-defined]
                _tooltip_html(label, probabilities.get(label), polish=polish), sticky=True
            ),
        ).add_to(fmap)
    note = (
        '<div style="position: fixed; top: 12px; left: 60px; z-index: 9999; '
        "background: #fcfcfb; color: #0b0b0b; padding: 8px 12px; "
        "border: 1px solid #c3c2b7; border-radius: 4px; "
        'font-family: system-ui, sans-serif; font-size: 13px;">'
        "Prediction abstained: no region reached the confidence threshold."
        "</div>"
    )
    fmap.get_root().html.add_child(folium_mod.Element(note))  # type: ignore[attr-defined]


def prediction_map(
    prediction: Prediction,
    *,
    top_k: int = 3,
    polish_labels: bool = False,
) -> folium.Map:
    """Render a prediction as an interactive folium map of Poland.

    The ``top_k`` most probable regions are drawn as circle markers whose
    radius and fill opacity scale with probability and whose colour is graded
    by rank (winner strongest). The geographic lookup table is auto-detected
    from ``prediction.level``. Abstained predictions render every catalogued
    region faintly plus an explanatory note.

    Args:
        prediction: The prediction to visualise.
        top_k: How many of the most probable regions to highlight.
        polish_labels: Use Polish display names in tooltips instead of English.

    Returns:
        A ``folium.Map`` centred on Poland, fitted to its bounding box.

    Raises:
        ConfigurationError: If ``top_k`` is less than 1.
        MissingDependencyError: If folium is not installed (extra ``viz``).
    """
    if top_k < 1:
        raise ConfigurationError(f"top_k must be >= 1, got {top_k}")
    folium_mod = optional_import("folium", extra="viz", purpose="prediction maps")
    fmap = _base_map(folium_mod)

    if prediction.abstained:
        _add_abstention_layer(folium_mod, fmap, prediction, polish=polish_labels)
        return fmap

    entries: list[tuple[str, float | None]] = [
        (cp.label, cp.probability) for cp in prediction.top_k(top_k)
    ]
    if not entries and prediction.label is not None:
        # A bare label without probabilities still deserves a marker.
        entries = [(prediction.label, None)]

    placed = 0
    for rank, (label, probability) in enumerate(entries):
        centroid = _centroid_for(label, prediction.level)
        if centroid is None:
            logger.warning(
                "no geometry for %s-level label %r; skipping it on the map",
                prediction.level.value,
                label,
            )
            continue
        weight_p = 1.0 if probability is None else probability
        color = _RANK_COLORS[min(rank, len(_RANK_COLORS) - 1)]
        folium_mod.CircleMarker(
            location=[centroid.lat, centroid.lon],
            radius=10.0 + 24.0 * weight_p,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.35 + 0.55 * weight_p,
            opacity=0.9,
            tooltip=folium_mod.Tooltip(
                _tooltip_html(label, probability, polish=polish_labels), sticky=True
            ),
        ).add_to(fmap)
        placed += 1
    if entries and placed == 0:
        logger.warning(
            "none of the top-%d predicted labels had known geometry; returning the base map",
            len(entries),
        )
    return fmap


def confidence_heatmap(prediction: Prediction, *, polish_labels: bool = False) -> folium.Map:
    """Shade every catalogued region by its predicted probability.

    Each region with known geometry gets a ~25 km ``folium.Circle`` whose
    colour and opacity follow the sequential blue ramp via :func:`_shade`
    (square-root perceptual mapping). Regions absent from the prediction's
    probabilities are shown at probability zero; predicted labels without
    geometry are logged and skipped.

    Args:
        prediction: The prediction to visualise.
        polish_labels: Use Polish display names in tooltips instead of English.

    Returns:
        A ``folium.Map`` centred on Poland, fitted to its bounding box.

    Raises:
        MissingDependencyError: If folium is not installed (extra ``viz``).
    """
    folium_mod = optional_import("folium", extra="viz", purpose="confidence heatmaps")
    fmap = _base_map(folium_mod)
    probabilities = {label.strip().lower(): p for label, p in prediction.as_dict().items()}
    table = _centroid_table(prediction.level)

    for label, centroid in table.items():
        probability = probabilities.get(label, 0.0)
        color, fill_opacity = _shade(probability)
        folium_mod.Circle(
            location=[centroid.lat, centroid.lon],
            radius=25_000.0,  # metres; ~25 km reads as a region-scale blob at country zoom
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            opacity=0.8,
            tooltip=folium_mod.Tooltip(
                _tooltip_html(label, probability, polish=polish_labels), sticky=True
            ),
        ).add_to(fmap)

    for label in probabilities:
        if label not in table:
            logger.warning(
                "no geometry for %s-level label %r; it is not shown on the heatmap",
                prediction.level.value,
                label,
            )
    return fmap


def save_map(fmap: folium.Map, path: str | Path) -> Path:
    """Write a folium map to ``path`` as a standalone UTF-8 HTML document.

    Args:
        fmap: The map to save.
        path: Destination file; parent directories are created as needed.

    Returns:
        The destination path as a :class:`~pathlib.Path`.
    """
    target = Path(path)
    ensure_dir(target.parent)
    fmap.save(str(target))
    logger.info("saved map to %s", target)
    return target
