"""Statistical power: the minimum detectable effect for a paired comparison.

The significance tests answer "is this gap real". Power answers the prior
question "could a gap this size even be seen at this n". On the small,
single-locality corpora tulip targets, n is often around 144, and a reader
should know that a two-point accuracy gap is below the noise floor before
reading anything into it.

This inverts the exact McNemar test. Two models are compared on the same
samples; the test looks only at the discordant pairs, where one model is right
and the other wrong. The most detectable case is a one-directional gap, where
every disagreement favours the better model. Under that model the number of net
wins is ``Binomial(n, delta)`` for a true accuracy gap ``delta``. The minimum
detectable effect is the smallest ``delta`` whose test rejects the null with the
requested power.

The floor reported here is optimistic on purpose: a realistic two-sided
disagreement needs a larger gap to reach the same power. Everything uses
``math.comb`` and stays exact, so no SciPy is added and the result is
deterministic.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from tulip.core.exceptions import ConfigurationError

__all__ = ["PowerReport", "minimum_detectable_effect"]

#: Cap on the discordant-count search; even a very small alpha needs few wins.
_MAX_DISCORDANTS = 200


class PowerReport(BaseModel):
    """The minimum detectable accuracy gap for a paired McNemar comparison."""

    model_config = ConfigDict(frozen=True)

    n_samples: int = Field(ge=1)
    alpha: float = Field(gt=0.0, lt=1.0)
    power: float = Field(gt=0.0, lt=1.0)
    significant_wins: int = Field(ge=1)
    detectable: bool
    mde: float | None = None

    def to_markdown(self) -> str:
        """Render the power result as one short markdown block."""
        from tulip.evaluation._format import format_metric

        title = "# Statistical power"
        if not self.detectable:
            body = (
                f"No accuracy gap is detectable at n={self.n_samples}: significance needs "
                f"at least {self.significant_wins} one-directional wins."
            )
        else:
            body = (
                f"Smallest one-directional accuracy gap detectable with {self.power:.0%} power "
                f"at alpha={self.alpha}, n={self.n_samples}: {format_metric(self.mde)} "
                f"(at least {self.significant_wins} net wins)."
            )
        return f"{title}\n\n{body}"


def minimum_detectable_effect(
    n_samples: int, *, alpha: float = 0.05, power: float = 0.8
) -> PowerReport:
    """Smallest paired accuracy gap detectable at this sample size.

    Args:
        n_samples: Number of paired samples the two models are scored on.
        alpha: Two-sided significance level.
        power: Target power (probability of detecting the effect).

    Returns:
        A :class:`PowerReport`. ``mde`` is ``None`` when no gap is detectable at
        this ``n``, because significance needs more one-directional wins than
        there are samples.

    Raises:
        ConfigurationError: if ``n_samples`` is below 1, or ``alpha``/``power``
            are outside ``(0, 1)``.
    """
    if n_samples < 1:
        raise ConfigurationError(f"n_samples must be >= 1, got {n_samples}")
    if not 0.0 < alpha < 1.0:
        raise ConfigurationError(f"alpha must be within (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ConfigurationError(f"power must be within (0, 1), got {power}")

    from tulip.evaluation.significance import _mcnemar_p

    wins = next(
        (m for m in range(1, _MAX_DISCORDANTS + 1) if _mcnemar_p(m, 0) <= alpha),
        _MAX_DISCORDANTS + 1,
    )
    detectable = wins <= n_samples
    mde = _search_mde(n_samples, wins, power) if detectable else None
    return PowerReport(
        n_samples=n_samples,
        alpha=alpha,
        power=power,
        significant_wins=wins,
        detectable=detectable,
        mde=mde,
    )


def _binomial_tail(n: int, k: int, p: float) -> float:
    """``P(Binomial(n, p) >= k)`` summed in log space.

    Beyond roughly a thousand samples ``math.comb(n, i)`` outgrows the float
    range, so forming ``comb * p**i`` directly raises OverflowError on the very
    split sizes a real benchmark reports on. Accumulating log-terms keeps every
    intermediate in range, and ``fsum`` keeps the many small addends accurate.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0  # at least one success is impossible
    if p >= 1.0:
        return 1.0  # every trial succeeds and k <= n
    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_n_factorial = math.lgamma(n + 1)
    terms = (
        math.exp(
            log_n_factorial
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
        for i in range(k, n + 1)
    )
    return min(1.0, math.fsum(terms))


def _search_mde(n: int, wins: int, power: float, *, tolerance: float = 1e-6) -> float:
    """Smallest ``delta`` with ``P(Binomial(n, delta) >= wins) >= power``.

    The tail probability rises monotonically with ``delta`` and reaches 1 at
    ``delta = 1`` (since ``wins <= n``), so a bisection converges.
    """
    low, high = 0.0, 1.0
    while high - low > tolerance:
        mid = (low + high) / 2.0
        if _binomial_tail(n, wins, mid) >= power:
            high = mid
        else:
            low = mid
    return high
