"""Tests for tulip.serve.settings.ServeSettings.from_env."""

from __future__ import annotations

from tulip.serve.settings import ServeSettings


class TestFromEnv:
    def test_defaults_when_unset(self) -> None:
        settings = ServeSettings.from_env({})
        assert settings.api_token is None
        assert settings.rate_limit_per_minute is None
        assert settings.max_concurrency is None
        assert settings.max_body_bytes == 32 * 1024 * 1024  # safe default, on
        assert settings.max_batch == 512
        assert settings.security_headers is True
        assert settings.cors_allow_origins == ()
        assert settings.auth_enabled is False

    def test_reads_prefixed_variables(self) -> None:
        settings = ServeSettings.from_env(
            {
                "TULIP_SERVE_API_TOKEN": "s3cret",
                "TULIP_SERVE_RATE_LIMIT": "120",
                "TULIP_SERVE_MAX_CONCURRENCY": "8",
                "TULIP_SERVE_MAX_BODY_BYTES": "1048576",
                "TULIP_SERVE_MAX_BATCH": "50",
                "TULIP_SERVE_CORS_ORIGINS": "http://a.test, http://b.test",
                "TULIP_SERVE_HSTS": "true",
            }
        )
        assert settings.api_token == "s3cret" and settings.auth_enabled
        assert settings.rate_limit_per_minute == 120
        assert settings.max_concurrency == 8
        assert settings.max_body_bytes == 1048576
        assert settings.max_batch == 50
        assert settings.cors_allow_origins == ("http://a.test", "http://b.test")
        assert settings.cors_enabled is True
        assert settings.hsts is True

    def test_non_positive_disables_a_limit(self) -> None:
        settings = ServeSettings.from_env({"TULIP_SERVE_MAX_BODY_BYTES": "0"})
        assert settings.max_body_bytes is None

    def test_blank_token_is_none(self) -> None:
        assert ServeSettings.from_env({"TULIP_SERVE_API_TOKEN": ""}).api_token is None

    def test_bool_parsing(self) -> None:
        assert ServeSettings.from_env({"TULIP_SERVE_SECURITY_HEADERS": "off"}).security_headers is (
            False
        )
        assert ServeSettings.from_env({"TULIP_SERVE_HSTS": "1"}).hsts is True

    def test_auth_exempt_paths_cover_liveness_and_metrics(self) -> None:
        assert set(ServeSettings().auth_exempt_paths) == {"/health", "/metrics"}
