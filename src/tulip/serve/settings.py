"""Environment-driven configuration for the serving guards.

The HTTP service is safe to bind to loopback out of the box, but before it faces
a network it needs the usual guards: authentication, rate limiting, a
concurrency cap, a request-body ceiling, CORS, and security headers. Those are
operational choices, not part of an experiment, so they are read from
``TULIP_SERVE_*`` environment variables into this frozen :class:`ServeSettings`
rather than bolted onto the frozen :class:`~tulip.config.schemas.ExperimentConfig`.

Defaults are chosen so the box is *safe by default without breaking loopback
use*: a generous body-size cap and security headers are on; auth, rate limiting,
concurrency limiting, and CORS are opt-in (a token/limit must be set to enable
them). No heavy import lives here, so ``import tulip.serve`` never requires
FastAPI just to read settings.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["ENV_PREFIX", "ServeSettings"]

#: Prefix for every environment variable this reads (``TULIP_SERVE_API_TOKEN`` …).
ENV_PREFIX = "TULIP_SERVE_"

#: Default request-body ceiling (32 MiB): comfortably above any real audio clip,
#: far below what a memory-exhaustion upload would need.
_DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024

#: Default per-call batch cap (matches the historical hard-coded value).
_DEFAULT_MAX_BATCH = 512

#: Paths exempt from authentication so liveness and scraping never need a token.
_AUTH_EXEMPT_PATHS = ("/health", "/metrics")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class ServeSettings(BaseModel):
    """Guard configuration for :func:`tulip.serve.app.create_app`.

    Attributes:
        api_token: Bearer token required on every request (except
            :attr:`auth_exempt_paths`) when set; ``None`` disables auth.
        rate_limit_per_minute: Per-client request ceiling per minute; ``None``
            disables rate limiting.
        max_concurrency: Maximum in-flight requests; ``None`` disables the cap.
        max_body_bytes: Request-body ceiling, enforced *before* buffering;
            ``None`` disables it.
        max_batch: Maximum texts per batch prediction call.
        cors_allow_origins: Allowed CORS origins; empty disables CORS.
        security_headers: Whether to add ``X-Content-Type-Options`` etc.
        hsts: Whether to add ``Strict-Transport-Security`` (HTTPS deployments).
    """

    model_config = ConfigDict(frozen=True)

    api_token: str | None = None
    rate_limit_per_minute: int | None = Field(default=None, gt=0)
    max_concurrency: int | None = Field(default=None, gt=0)
    max_body_bytes: int | None = Field(default=_DEFAULT_MAX_BODY_BYTES, gt=0)
    max_batch: int = Field(default=_DEFAULT_MAX_BATCH, gt=0)
    cors_allow_origins: tuple[str, ...] = ()
    security_headers: bool = True
    hsts: bool = False

    @property
    def auth_exempt_paths(self) -> tuple[str, ...]:
        """Paths that never require the bearer token (liveness, metrics)."""
        return _AUTH_EXEMPT_PATHS

    @property
    def auth_enabled(self) -> bool:
        return self.api_token is not None

    @property
    def cors_enabled(self) -> bool:
        return bool(self.cors_allow_origins)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServeSettings:
        """Build settings from ``TULIP_SERVE_*`` variables (``os.environ`` by default).

        Unset variables fall back to the field defaults. A body-size or limit
        variable set to a non-positive value disables that guard (becomes
        ``None``); booleans accept ``1/true/yes/on`` (case-insensitive).

        Args:
            environ: Variable source; defaults to :data:`os.environ`. Injectable
                for testing.

        Returns:
            The parsed, validated settings.
        """
        env = os.environ if environ is None else environ
        return cls(
            api_token=env.get(f"{ENV_PREFIX}API_TOKEN") or None,
            rate_limit_per_minute=_positive_or_none(env.get(f"{ENV_PREFIX}RATE_LIMIT")),
            max_concurrency=_positive_or_none(env.get(f"{ENV_PREFIX}MAX_CONCURRENCY")),
            max_body_bytes=_positive_or_none(
                env.get(f"{ENV_PREFIX}MAX_BODY_BYTES"), default=_DEFAULT_MAX_BODY_BYTES
            ),
            max_batch=_positive_or_default(env.get(f"{ENV_PREFIX}MAX_BATCH"), _DEFAULT_MAX_BATCH),
            cors_allow_origins=_csv(env.get(f"{ENV_PREFIX}CORS_ORIGINS")),
            security_headers=_flag(env.get(f"{ENV_PREFIX}SECURITY_HEADERS"), default=True),
            hsts=_flag(env.get(f"{ENV_PREFIX}HSTS"), default=False),
        )


def _positive_or_none(value: str | None, *, default: int | None = None) -> int | None:
    """Parse a positive int; a missing value yields ``default``, ``<= 0`` yields ``None``."""
    if value is None or not value.strip():
        return default
    parsed = int(value)
    return parsed if parsed > 0 else None


def _positive_or_default(value: str | None, default: int) -> int:
    """Parse a positive int, falling back to ``default`` when unset/non-positive."""
    if value is None or not value.strip():
        return default
    parsed = int(value)
    return parsed if parsed > 0 else default


def _csv(value: str | None) -> tuple[str, ...]:
    """Split a comma-separated variable into a tuple of trimmed, non-empty items."""
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _flag(value: str | None, *, default: bool) -> bool:
    """Parse a boolean flag (``1/true/yes/on``); unset yields ``default``."""
    if value is None or not value.strip():
        return default
    return value.strip().lower() in _TRUTHY
