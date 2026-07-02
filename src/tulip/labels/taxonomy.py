"""Hierarchical taxonomy of Polish dialects.

The taxonomy follows the traditional dialectological classification
(Nitsch/Urbanczyk) of Polish into major dialect groups ("families"), each
containing regional dialects (gwary). Kashubian is treated as its own
top-level group, reflecting its status as a distinct ethnolect, and
``STANDARD`` covers standard (general) Polish, which serves as the negative
class when deciding whether a sample is dialectal at all.

Labels are hierarchical: village -> region -> regional dialect -> family,
plus the administrative voivodeship. Models may be trained at any level via
:class:`LabelLevel`, and hierarchical classifiers can back off from
fine-grained to coarse-grained predictions.
"""

from __future__ import annotations

import enum


class LabelLevel(str, enum.Enum):
    """Granularity at which a classifier is trained and evaluated."""

    FAMILY = "family"
    DIALECT = "dialect"
    REGION = "region"
    VILLAGE = "village"
    VOIVODESHIP = "voivodeship"


class DialectFamily(str, enum.Enum):
    """Major Polish dialect groups."""

    GREATER_POLISH = "greater_polish"
    LESSER_POLISH = "lesser_polish"
    MASOVIAN = "masovian"
    SILESIAN = "silesian"
    KASHUBIAN = "kashubian"
    STANDARD = "standard"


class RegionalDialect(str, enum.Enum):
    """Regional dialects (gwary) recognised by the toolkit.

    The set is intentionally extensible: corpora may carry labels outside this
    enum (as plain strings) and still flow through the pipeline. The enum
    captures the regions with dedicated resources and map geometry.
    """

    KURPIE = "kurpie"
    PODHALE = "podhale"
    SPISZ = "spisz"
    ORAWA = "orawa"
    KASHUBIA = "kashubia"
    KOCIEWIE = "kociewie"
    KUJAWY = "kujawy"
    WARMIA = "warmia"
    MASURIA = "masuria"
    PODLASIE = "podlasie"
    SILESIA = "silesia"
    CIESZYN_SILESIA = "cieszyn_silesia"
    GREATER_POLAND = "greater_poland"
    LESSER_POLAND = "lesser_poland"
    MAZOVIA_PROPER = "mazovia_proper"
    PODOLIA = "podolia"  # borderland corpus of Mackowce (Kresy)


#: Mapping from each regional dialect to its dialect family.
REGION_TO_FAMILY: dict[RegionalDialect, DialectFamily] = {
    RegionalDialect.KURPIE: DialectFamily.MASOVIAN,
    RegionalDialect.PODHALE: DialectFamily.LESSER_POLISH,
    RegionalDialect.SPISZ: DialectFamily.LESSER_POLISH,
    RegionalDialect.ORAWA: DialectFamily.LESSER_POLISH,
    RegionalDialect.KASHUBIA: DialectFamily.KASHUBIAN,
    RegionalDialect.KOCIEWIE: DialectFamily.GREATER_POLISH,
    RegionalDialect.KUJAWY: DialectFamily.GREATER_POLISH,
    RegionalDialect.WARMIA: DialectFamily.MASOVIAN,
    RegionalDialect.MASURIA: DialectFamily.MASOVIAN,
    RegionalDialect.PODLASIE: DialectFamily.MASOVIAN,
    RegionalDialect.SILESIA: DialectFamily.SILESIAN,
    RegionalDialect.CIESZYN_SILESIA: DialectFamily.SILESIAN,
    RegionalDialect.GREATER_POLAND: DialectFamily.GREATER_POLISH,
    RegionalDialect.LESSER_POLAND: DialectFamily.LESSER_POLISH,
    RegionalDialect.MAZOVIA_PROPER: DialectFamily.MASOVIAN,
    RegionalDialect.PODOLIA: DialectFamily.LESSER_POLISH,
}

_DISPLAY_NAMES: dict[str, tuple[str, str]] = {
    # value -> (English, Polish)
    DialectFamily.GREATER_POLISH.value: ("Greater Polish", "dialekt wielkopolski"),
    DialectFamily.LESSER_POLISH.value: ("Lesser Polish", "dialekt malopolski"),
    DialectFamily.MASOVIAN.value: ("Masovian", "dialekt mazowiecki"),
    DialectFamily.SILESIAN.value: ("Silesian", "dialekt slaski"),
    DialectFamily.KASHUBIAN.value: ("Kashubian", "kaszubski"),
    DialectFamily.STANDARD.value: ("Standard Polish", "polszczyzna ogolna"),
    RegionalDialect.KURPIE.value: ("Kurpie", "gwara kurpiowska"),
    RegionalDialect.PODHALE.value: ("Podhale", "gwara podhalanska"),
    RegionalDialect.SPISZ.value: ("Spisz", "gwara spiska"),
    RegionalDialect.ORAWA.value: ("Orawa", "gwara orawska"),
    RegionalDialect.KASHUBIA.value: ("Kashubia", "kaszubski"),
    RegionalDialect.KOCIEWIE.value: ("Kociewie", "gwara kociewska"),
    RegionalDialect.KUJAWY.value: ("Kujawy", "gwara kujawska"),
    RegionalDialect.WARMIA.value: ("Warmia", "gwara warminska"),
    RegionalDialect.MASURIA.value: ("Masuria", "gwara mazurska"),
    RegionalDialect.PODLASIE.value: ("Podlasie", "gwara podlaska"),
    RegionalDialect.SILESIA.value: ("Silesia", "gwara slaska"),
    RegionalDialect.CIESZYN_SILESIA.value: ("Cieszyn Silesia", "gwara cieszynska"),
    RegionalDialect.GREATER_POLAND.value: ("Greater Poland", "gwary wielkopolskie"),
    RegionalDialect.LESSER_POLAND.value: ("Lesser Poland", "gwary malopolskie"),
    RegionalDialect.MAZOVIA_PROPER.value: ("Mazovia", "gwary mazowieckie"),
    RegionalDialect.PODOLIA.value: ("Podolia (Mackowce)", "gwara mackowiecka"),
}


def family_for(dialect: RegionalDialect | str) -> DialectFamily | None:
    """Return the dialect family for a regional dialect, or ``None`` if unknown."""
    if isinstance(dialect, str):
        try:
            dialect = RegionalDialect(dialect.strip().lower())
        except ValueError:
            return None
    return REGION_TO_FAMILY.get(dialect)


def display_name(label: str | enum.Enum, *, polish: bool = False) -> str:
    """Return a human-readable name for a taxonomy label.

    Falls back to title-casing the raw value for labels outside the taxonomy,
    so corpus-specific labels still render reasonably in reports and maps.
    """
    value = label.value if isinstance(label, enum.Enum) else str(label)
    names = _DISPLAY_NAMES.get(value.strip().lower())
    if names is None:
        return value.replace("_", " ").title()
    return names[1] if polish else names[0]
