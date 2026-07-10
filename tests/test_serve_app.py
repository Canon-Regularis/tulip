"""Tests for the FastAPI inference service (optional extra ``serve``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def client(trained_text_artifact: Path) -> TestClient:
    from tulip.serve.app import create_app

    return TestClient(create_app(trained_text_artifact))


class TestUndecodableAudioUpload:
    """A `.wav` name proves nothing about the bytes inside it."""

    def test_undecodable_audio_is_a_400_not_a_500(self, monkeypatch, tmp_path: Path) -> None:
        from tulip.core.exceptions import DataError
        from tulip.core.types import TaskType
        from tulip.pipeline import DialectClassifier

        class _AudioStub:
            task = TaskType.AUDIO
            classes_ = ("podhale", "silesia")
            target = None

            def predict(self, raw: object) -> object:
                raise DataError(f"could not decode audio file {raw}")

        monkeypatch.setattr(DialectClassifier, "load", classmethod(lambda cls, path: _AudioStub()))
        from tulip.serve.app import create_app

        client = TestClient(create_app(tmp_path))
        response = client.post("/predict/audio", files={"file": ("x.wav", b"definitely not audio")})

        assert response.status_code == 400, response.text
        assert "could not be decoded" in response.json()["detail"]


class TestHealth:
    def test_health_reports_model_identity(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["task"] == "text"
        assert set(payload["classes"]) == {"podhale", "silesia", "kurpie"}


class TestPredictText:
    def test_returns_full_prediction_json(self, client: TestClient) -> None:
        response = client.post(
            "/predict/text", json={"text": "Hej, baca się pyto, kaj się owce pasą na holi."}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["label"] == "podhale"
        assert payload["abstained"] is False
        probabilities = [entry["probability"] for entry in payload["probabilities"]]
        assert probabilities == sorted(probabilities, reverse=True)
        assert sum(probabilities) == pytest.approx(1.0)

    def test_top_k_truncates_distribution(self, client: TestClient) -> None:
        response = client.post("/predict/text", json={"text": "Godom po naszymu.", "top_k": 1})
        assert response.status_code == 200
        assert len(response.json()["probabilities"]) == 1

    def test_empty_text_is_rejected(self, client: TestClient) -> None:
        assert client.post("/predict/text", json={"text": ""}).status_code == 422
        assert client.post("/predict/text", json={"text": "   "}).status_code == 400

    def test_malformed_body_is_rejected(self, client: TestClient) -> None:
        assert client.post("/predict/text", json={"tekst": "zła nazwa pola"}).status_code == 422


class TestPredictAudio:
    def test_text_model_rejects_audio_uploads(self, client: TestClient) -> None:
        response = client.post(
            "/predict/audio", files={"file": ("clip.wav", b"RIFF....", "audio/wav")}
        )
        assert response.status_code == 400
        assert "not audio" in response.json()["detail"]


class TestCreateApp:
    def test_missing_artifact_raises_data_error(self, tmp_path: Path) -> None:
        from tulip.core.exceptions import DataError
        from tulip.serve.app import create_app

        with pytest.raises(DataError):
            create_app(tmp_path / "missing")
