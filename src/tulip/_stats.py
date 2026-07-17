"""Small, dependency-free statistics shared by the evaluation and explanation layers.

Three reports rest on the same two textbook procedures. The significance report
pairs models, the fairness report compares subgroups, and the contrastive analysis
compares two dialects; each needs an unpaired two-proportion z-test and a
Holm-Bonferroni correction over a family of p-values. One copy lives here, at the
package root beside the other shared leaf helpers, so the three cannot drift apart.
Everything is pure ``math``, so any layer imports it without risking a cycle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["holm_correct", "two_proportion_p"]


def two_proportion_p(count_a: int, n_a: int, count_b: int, n_b: int) -> float:
    """Two-sided unpaired two-proportion z-test p-value (normal tail via ``erfc``).

    Returns ``1.0`` for a degenerate comparison, an empty group or no variation to
    separate the two, so a caller reads "not significant" rather than dividing by
    zero.
    """
    if n_a == 0 or n_b == 0:
        return 1.0
    pooled = (count_a + count_b) / (n_a + n_b)
    if pooled in (0.0, 1.0):
        return 1.0  # no variation to separate the groups
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if standard_error == 0.0:
        return 1.0
    z = (count_a / n_a - count_b / n_b) / standard_error
    return math.erfc(abs(z) / math.sqrt(2.0))


def holm_correct(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment of ``p_values``, input order preserved.

    Each p-value is scaled by the number of hypotheses not yet rejected and the
    sequence is kept monotone, so testing a whole family does not inflate the
    family-wise error rate. Returns the adjusted values in the original order; an
    empty input returns ``[]``.
    """
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
