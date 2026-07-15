"""Approximate geography for Polish dialect regions and voivodeships.

Coordinates are representative centroids (WGS84 lat/lon) intended for map
visualisation: highlighting a predicted region, drawing confidence heatmaps,
and placing markers. They are deliberately approximate; dialect boundaries are
gradients, not lines.
"""

from __future__ import annotations

from typing import NamedTuple

from tulip.labels.taxonomy import RegionalDialect


class GeoPoint(NamedTuple):
    """A WGS84 coordinate pair."""

    lat: float
    lon: float


#: Geographic centre of Poland, used to initialise map views.
POLAND_CENTER = GeoPoint(52.06, 19.48)

#: (south, west, north, east) bounding box of Poland.
POLAND_BOUNDS = (49.0, 14.12, 54.84, 24.15)

#: Representative centroid of each regional dialect area.
REGION_CENTROIDS: dict[RegionalDialect, GeoPoint] = {
    RegionalDialect.KURPIE: GeoPoint(53.25, 21.35),
    RegionalDialect.PODHALE: GeoPoint(49.35, 19.95),
    RegionalDialect.SPISZ: GeoPoint(49.42, 20.42),
    RegionalDialect.ORAWA: GeoPoint(49.52, 19.72),
    RegionalDialect.KASHUBIA: GeoPoint(54.25, 18.00),
    RegionalDialect.KOCIEWIE: GeoPoint(53.95, 18.55),
    RegionalDialect.KUJAWY: GeoPoint(52.75, 18.55),
    RegionalDialect.WARMIA: GeoPoint(53.95, 20.45),
    RegionalDialect.MASURIA: GeoPoint(53.85, 21.55),
    RegionalDialect.PODLASIE: GeoPoint(53.10, 23.00),
    RegionalDialect.SILESIA: GeoPoint(50.30, 18.70),
    RegionalDialect.CIESZYN_SILESIA: GeoPoint(49.75, 18.63),
    RegionalDialect.GREATER_POLAND: GeoPoint(52.30, 17.30),
    RegionalDialect.LESSER_POLAND: GeoPoint(50.05, 20.50),
    RegionalDialect.MAZOVIA_PROPER: GeoPoint(52.40, 21.00),
    # Historical Podolia lies outside modern Poland (Mackowce, Ukraine).
    RegionalDialect.PODOLIA: GeoPoint(49.25, 27.35),
}

#: Representative centroid of each of the 16 Polish voivodeships.
VOIVODESHIP_CENTROIDS: dict[str, GeoPoint] = {
    "dolnoslaskie": GeoPoint(51.10, 16.40),
    "kujawsko-pomorskie": GeoPoint(53.05, 18.50),
    "lubelskie": GeoPoint(51.20, 22.90),
    "lubuskie": GeoPoint(52.20, 15.30),
    "lodzkie": GeoPoint(51.60, 19.40),
    "malopolskie": GeoPoint(49.90, 20.40),
    "mazowieckie": GeoPoint(52.35, 21.10),
    "opolskie": GeoPoint(50.60, 17.90),
    "podkarpackie": GeoPoint(50.05, 22.00),
    "podlaskie": GeoPoint(53.30, 23.00),
    "pomorskie": GeoPoint(54.20, 18.00),
    "slaskie": GeoPoint(50.30, 19.00),
    "swietokrzyskie": GeoPoint(50.80, 20.80),
    "warminsko-mazurskie": GeoPoint(53.90, 20.80),
    "wielkopolskie": GeoPoint(52.30, 17.30),
    "zachodniopomorskie": GeoPoint(53.60, 15.50),
}


def region_centroid(region: RegionalDialect | str) -> GeoPoint | None:
    """Return the centroid for a regional dialect, or ``None`` if unknown."""
    if isinstance(region, str):
        try:
            region = RegionalDialect(region.strip().lower())
        except ValueError:
            return None
    return REGION_CENTROIDS.get(region)


def voivodeship_centroid(name: str) -> GeoPoint | None:
    """Return the centroid for a voivodeship by (diacritic-free) name."""
    return VOIVODESHIP_CENTROIDS.get(name.strip().lower())
