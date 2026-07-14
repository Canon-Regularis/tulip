"""Tests for the serving guards (auth, rate limit, concurrency, body size, headers)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from tulip.serve._guards import (  # noqa: E402
    BodySizeLimitMiddleware,
    ConcurrencyLimitMiddleware,
)
from tulip.serve.settings import ServeSettings  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

_TEXT = "Hej, baca się pyto, kaj się owce pasą na holi."


def _client(artifact: Path, **settings: Any) -> TestClient:
    from tulip.serve.app import create_app

    return TestClient(create_app(artifact, settings=ServeSettings(**settings)))


class TestSecurityHeaders:
    def test_present_by_default(self, trained_text_artifact: Path) -> None:
        response = _client(trained_text_artifact).get("/health")
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "content-security-policy" in response.headers

    def test_can_be_disabled(self, trained_text_artifact: Path) -> None:
        response = _client(trained_text_artifact, security_headers=False).get("/health")
        assert "x-content-type-options" not in response.headers


class TestAuth:
    def test_rejects_without_token(self, trained_text_artifact: Path) -> None:
        response = _client(trained_text_artifact, api_token="secret").post(
            "/predict/text", json={"text": _TEXT}
        )
        assert response.status_code == 401

    def test_accepts_valid_token(self, trained_text_artifact: Path) -> None:
        response = _client(trained_text_artifact, api_token="secret").post(
            "/predict/text", json={"text": _TEXT}, headers={"Authorization": "Bearer secret"}
        )
        assert response.status_code == 200

    def test_health_is_exempt(self, trained_text_artifact: Path) -> None:
        response = _client(trained_text_artifact, api_token="secret").get("/health")
        assert response.status_code == 200


class TestBodySizeLimit:
    def test_oversized_content_length_is_413(self, trained_text_artifact: Path) -> None:
        response = _client(trained_text_artifact, max_body_bytes=10).post(
            "/predict/text",
            json={"text": _TEXT},  # body is well over 10 bytes
        )
        assert response.status_code == 413

    def test_within_limit_passes(self, trained_text_artifact: Path) -> None:
        response = _client(trained_text_artifact, max_body_bytes=10_000).post(
            "/predict/text", json={"text": _TEXT}
        )
        assert response.status_code == 200


class TestCors:
    def test_allowed_origin_is_reflected(self, trained_text_artifact: Path) -> None:
        response = _client(trained_text_artifact, cors_allow_origins=("http://a.test",)).get(
            "/health", headers={"Origin": "http://a.test"}
        )
        assert response.headers.get("access-control-allow-origin") == "http://a.test"


class TestRateLimit:
    def test_second_request_is_throttled(self, trained_text_artifact: Path) -> None:
        client = _client(trained_text_artifact, rate_limit_per_minute=1)
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 429  # bucket empty within the minute


class TestObservabilityStillCountsRejects:
    def test_a_rejected_request_is_metered(self, trained_text_artifact: Path) -> None:
        client = _client(trained_text_artifact, api_token="secret")
        client.post("/predict/text", json={"text": _TEXT})  # 401, inner to observability
        metrics = client.get("/metrics").text
        assert 'status="401"' in metrics  # the reject was counted


class TestModelIdentityHeaders:
    def test_version_and_digest_headers_when_provided(self, trained_text_artifact: Path) -> None:
        from tulip.serve.app import create_app

        client = TestClient(
            create_app(trained_text_artifact, model_version="3", model_digest="abc123")
        )
        response = client.post("/predict/text", json={"text": _TEXT})
        assert response.headers["x-model-version"] == "3"
        assert response.headers["x-model-digest"] == "abc123"


# ----------------------------- ASGI-level unit tests -----------------------------


async def _drive(middleware: Any, scope: dict[str, Any], body: bytes = b"") -> list[dict[str, Any]]:
    """Drive an ASGI middleware once and return the messages it sent."""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


async def _ok_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    await receive()  # read the body (this is what triggers body-size counting)
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def _status(sent: list[dict[str, Any]]) -> int:
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_concurrency_rejects_when_full() -> None:
    middleware = ConcurrencyLimitMiddleware(_ok_app, max_concurrency=1)
    middleware._in_flight = 1  # simulate one request already in flight
    sent = asyncio.run(_drive(middleware, {"type": "http", "path": "/x"}))
    assert _status(sent) == 503


def test_streamed_body_over_limit_is_413() -> None:
    # No Content-Length header: the guard must still stop an oversized stream.
    middleware = BodySizeLimitMiddleware(_ok_app, max_bytes=4)
    sent = asyncio.run(_drive(middleware, {"type": "http", "headers": []}, body=b"0123456789"))
    assert _status(sent) == 413
