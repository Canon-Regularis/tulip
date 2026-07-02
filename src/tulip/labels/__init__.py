"""Label taxonomy and geography for Polish dialects.

This package is dependency-free (standard library only) so that any other
tulip module can import it without pulling in heavy dependencies.
"""

from tulip.labels.taxonomy import (
    REGION_TO_FAMILY,
    DialectFamily,
    LabelLevel,
    RegionalDialect,
    display_name,
    family_for,
)

__all__ = [
    "REGION_TO_FAMILY",
    "DialectFamily",
    "LabelLevel",
    "RegionalDialect",
    "display_name",
    "family_for",
]
