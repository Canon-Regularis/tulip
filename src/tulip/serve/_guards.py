"""Optional serving guards: auth, rate limit, concurrency, body size, headers.

These are the guards the service needs before it faces a network, installed from
:class:`~tulip.serve.settings.ServeSettings` by :func:`install_guards`. Each is a
small, single-purpose ASGI middleware, so an operator enables exactly the ones
they want and the rest add nothing.

**Ordering.** :func:`install_guards` runs *before* the app's observability
middleware, which therefore stays the outermost layer: a request rejected by any
guard (401/429/503/413) is still timed, counted, and logged, and still carries
the security and CORS headers, because those layers wrap the auth/limit layers.
The intended nesting, outermost first::

    observability -> security-headers -> CORS -> auth -> rate-limit
        -> concurrency -> body-size -> handler

CORS is not reimplemented -- Starlette's :class:`CORSMiddleware` is reused, and
it sits outside auth so a browser preflight is answered without a token.

**The body-size guard is the load-bearing one.** ``POST /predict/audio`` reads
the whole upload into memory; without a ceiling an attacker exhausts RAM with one
request. :class:`BodySizeLimitMiddleware` rejects an oversized ``Content-Length``
before a byte is buffered and also counts a chunked body as it streams, so the
limit holds even when the length is not declared up front.

Rate-limit and concurrency state is per-process: under multiple workers each
worker enforces the limit independently (documented, not a bug).
"""

from __future__ import annotations

import hmac
import json
import time
from typing import TYPE_CHECKING, Any

from tulip.serve.settings import ServeSettings
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    Scope = dict[str, Any]
    Receive = Callable[[], Awaitable[dict[str, Any]]]
    Send = Callable[[dict[str, Any]], Awaitable[None]]

__all__ = [
    "AuthMiddleware",
    "BodySizeLimitMiddleware",
    "ConcurrencyLimitMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
    "install_guards",
]

logger = get_logger(__name__)


async def _send_json(send: Send, status: int, detail: str) -> None:
    """Send a minimal JSON error response over the raw ASGI channel.

    Shared by every guard so the reject-response boilerplate exists once.
    """
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _header(scope: Scope, name: bytes) -> str | None:
    """Return one request header value (decoded), or ``None`` if absent."""
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes`` before buffering them."""

    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _header(scope, b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await _send_json(send, 413, "request body too large")
            return

        received = 0
        started = False

        async def counting_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: dict[str, Any]) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLargeError:
            # If the handler had not started responding, answer cleanly; otherwise
            # the response is already in flight and only aborting is possible.
            if not started:
                await _send_json(send, 413, "request body too large")


class _BodyTooLargeError(Exception):
    """Internal signal that a streamed body exceeded the limit."""


class AuthMiddleware:
    """Require a bearer token on every request except the exempt paths."""

    def __init__(self, app: Any, *, token: str, exempt_paths: Iterable[str]) -> None:
        self.app = app
        self._expected = f"Bearer {token}"
        self._exempt = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self._exempt:
            await self.app(scope, receive, send)
            return
        provided = _header(scope, b"authorization") or ""
        # Constant-time compare so a wrong token cannot be guessed byte-by-byte
        # from response timing.
        if not hmac.compare_digest(provided, self._expected):
            await _send_json(send, 401, "missing or invalid bearer token")
            return
        await self.app(scope, receive, send)


class RateLimitMiddleware:
    """Per-client token-bucket rate limit (``per_minute`` requests, refilled continuously)."""

    def __init__(self, app: Any, *, per_minute: int) -> None:
        self.app = app
        self.capacity = float(per_minute)
        self.refill_per_second = per_minute / 60.0
        # client -> (tokens, last_seen_monotonic). No lock needed: the update runs
        # to completion between awaits on the single-threaded event loop.
        self._buckets: dict[str, tuple[float, float]] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if not self._take(_client(scope)):
            await _send_json(send, 429, "rate limit exceeded")
            return
        await self.app(scope, receive, send)

    def _take(self, client: str) -> bool:
        now = time.monotonic()
        tokens, last = self._buckets.get(client, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)
        if tokens < 1.0:
            self._buckets[client] = (tokens, now)
            return False
        self._buckets[client] = (tokens - 1.0, now)
        return True


class ConcurrencyLimitMiddleware:
    """Reject requests once ``max_concurrency`` are already in flight."""

    def __init__(self, app: Any, *, max_concurrency: int) -> None:
        self.app = app
        self.max = max_concurrency
        self._in_flight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if self._in_flight >= self.max:
            await _send_json(send, 503, "server at capacity")
            return
        self._in_flight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self._in_flight -= 1


class SecurityHeadersMiddleware:
    """Add standard security headers to every response."""

    def __init__(self, app: Any, *, hsts: bool) -> None:
        self.app = app
        headers = [
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
            # The demo page is self-contained (inline CSS/JS, no external hosts),
            # so 'unsafe-inline' is required for it while every remote origin stays
            # blocked; JSON API responses are unaffected by CSP.
            (
                b"content-security-policy",
                b"default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                b"script-src 'self' 'unsafe-inline'; base-uri 'none'; form-action 'self'",
            ),
        ]
        if hsts:
            headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
        self._headers = headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + self._headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _client(scope: Scope) -> str:
    """The client host used to key rate limiting (``unknown`` if unavailable)."""
    client = scope.get("client")
    return client[0] if client else "unknown"


def install_guards(app: Any, settings: ServeSettings) -> None:
    """Install the enabled guards on ``app`` in the documented nesting order.

    Call this *before* the app's observability middleware so observability
    remains outermost. Each guard is added only when its setting enables it;
    body-size and security headers are on by default.

    Args:
        app: The FastAPI/Starlette application.
        settings: The parsed serving settings.
    """
    # Added inner-to-outer: ``add_middleware`` prepends, so the last added wraps
    # outermost. See the module docstring for the resulting order.
    if settings.max_body_bytes is not None:
        app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    if settings.max_concurrency is not None:
        app.add_middleware(ConcurrencyLimitMiddleware, max_concurrency=settings.max_concurrency)
    if settings.rate_limit_per_minute is not None:
        app.add_middleware(RateLimitMiddleware, per_minute=settings.rate_limit_per_minute)
    if settings.auth_enabled:
        assert settings.api_token is not None  # noqa: S101 - narrowed by auth_enabled
        app.add_middleware(
            AuthMiddleware, token=settings.api_token, exempt_paths=settings.auth_exempt_paths
        )
    if settings.cors_enabled:
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    if settings.security_headers:
        app.add_middleware(SecurityHeadersMiddleware, hsts=settings.hsts)
    logger.debug(
        "serve guards: auth=%s rate_limit=%s concurrency=%s body_limit=%s cors=%s headers=%s",
        settings.auth_enabled,
        settings.rate_limit_per_minute,
        settings.max_concurrency,
        settings.max_body_bytes,
        settings.cors_enabled,
        settings.security_headers,
    )
