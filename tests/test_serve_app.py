"""Tests for the FastAPI inference service (optional extra ``serve``)."""

from __future__ import annotations

import re
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


class TestObservabilityHardening:
    """Security + robustness fixes for the observability layer (quality pass 2)."""

    def test_unmatched_paths_do_not_explode_metric_cardinality(self, client: TestClient) -> None:
        """A 404 flood must not add a metric series per attacker-controlled path.

        The raw path used to become the metric label, so hitting distinct URLs
        grew the registry without bound -- an unauthenticated memory DoS.
        """
        registry = client.app.state.metrics
        for i in range(150):
            client.get(f"/no-such-path-{i}")
        paths = {key[1] for key in registry._request_counts}
        assert "<unmatched>" in paths
        assert not any(p.startswith("/no-such-path-") for p in paths)
        assert "no-such-path" not in client.get("/metrics").text

    def test_unknown_http_method_is_bucketed(self, client: TestClient) -> None:
        client.request("PROPFIND", "/predict/text")
        methods = {key[0] for key in client.app.state.metrics._request_counts}
        assert "OTHER" in methods
        assert "PROPFIND" not in methods

    def test_unhandled_500_is_headed_counted_and_clean(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unhandled handler error must not bypass observability.

        It is still timed, counted as a 500, and answered with a clean JSON body
        carrying the correlation-ID/timing headers -- not an unheaded, uncounted,
        unlogged failure.
        """
        from tulip.pipeline import DialectClassifier

        def _boom(self: object, raw: object) -> object:
            raise RuntimeError("boom")

        monkeypatch.setattr(DialectClassifier, "predict", _boom)
        # TestClient re-raises server exceptions by default; opt out to observe the response.
        raw_client = TestClient(client.app, raise_server_exceptions=False)
        response = raw_client.post("/predict/text", json={"text": "hej baca"})

        assert response.status_code == 500
        assert response.headers["X-Request-ID"]
        assert float(response.headers["X-Process-Time-Ms"]) >= 0.0
        assert response.json() == {"detail": "internal server error"}
        assert any(key[2] == 500 for key in client.app.state.metrics._request_counts)


class TestDemoNoInjection:
    """The demo must not build markup from model class labels via innerHTML (XSS)."""

    def test_labels_are_rendered_via_textcontent_not_innerhtml(self) -> None:
        from tulip.serve._demo import demo_page

        page = demo_page(title="tulip")
        # The bug: row.innerHTML built from prettify(p.label). The fix uses
        # textContent/.title, which the browser escapes automatically.
        assert "row.innerHTML" not in page
        assert "bars.innerHTML" not in page
        assert ".textContent = prettify(p.label)" in page


class TestCreateApp:
    def test_missing_artifact_raises_data_error(self, tmp_path: Path) -> None:
        from tulip.core.exceptions import DataError
        from tulip.serve.app import create_app

        with pytest.raises(DataError):
            create_app(tmp_path / "missing")


_PODHALE = "Hej, baca się pyto, kaj się owce pasą na holi."
_SILESIA = "Jo żech je z Katowic i godom po naszymu cołki czos."
_KURPIE = "U nos w boru to psiwo warzą jesce po staremu."


def _total_requests(exposition: str) -> int:
    """Sum every ``tulip_requests_total`` series in Prometheus exposition text."""
    return sum(
        int(value)
        for value in re.findall(r"^tulip_requests_total\{[^}]*\} (\d+)$", exposition, re.MULTILINE)
    )


class TestObservabilityMiddleware:
    def test_predict_stamps_request_id_and_timing_headers(self, client: TestClient) -> None:
        response = client.post("/predict/text", json={"text": _PODHALE})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"]
        # X-Process-Time-Ms is a parseable, non-negative duration.
        assert float(response.headers["X-Process-Time-Ms"]) >= 0.0

    def test_supplied_request_id_is_echoed_unchanged(self, client: TestClient) -> None:
        response = client.post(
            "/predict/text", json={"text": _PODHALE}, headers={"X-Request-ID": "trace-abc-123"}
        )
        assert response.headers["X-Request-ID"] == "trace-abc-123"


class TestMetrics:
    def test_exposition_is_prometheus_text_and_counts_grow(self, client: TestClient) -> None:
        client.post("/predict/text", json={"text": _PODHALE})
        before = _total_requests(client.get("/metrics").text)
        client.get("/health")

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        assert "tulip_requests_total" in body
        assert "# TYPE tulip_requests_total counter" in body
        assert "# TYPE tulip_request_duration_ms summary" in body
        assert "tulip_request_duration_ms_sum" in body
        assert "tulip_request_duration_ms_count" in body
        # The extra /health and /metrics calls above must have been recorded.
        assert _total_requests(body) > before


class TestModelIdentityHeaders:
    def test_predict_text_carries_version_and_target(self, client: TestClient) -> None:
        from tulip import __version__

        response = client.post("/predict/text", json={"text": _PODHALE})
        assert response.headers["X-Tulip-Version"] == __version__
        assert response.headers["X-Model-Target"] == "dialect"
        assert "podhale" in response.headers["X-Model-Classes"].split(",")

    def test_polish_class_labels_do_not_crash_the_response(self, tmp_path: Path) -> None:
        """Diacritic class labels used to 500 every /predict via a latin-1 header.

        This is a Polish dialect toolkit, so labels with diacritics (śląsk,
        kaszëby) are the norm, not an edge case; the identity header must encode
        them safely instead of raising UnicodeEncodeError.
        """
        from urllib.parse import unquote

        from tulip.core.types import DialectLabels, Sample
        from tulip.pipeline import DialectClassifier
        from tulip.serve.app import create_app

        samples = [
            Sample(
                id=str(i),
                text=f"godom po naszymu tekst numer {i}",
                speaker_id=f"spk{i % 4}",
                labels=DialectLabels(dialect="śląsk" if i % 2 else "kaszëby"),
                source="t",
            )
            for i in range(12)
        ]
        model_dir = tmp_path / "m"
        DialectClassifier(model="logistic_regression", features=["char_tfidf"]).fit(samples).save(
            model_dir
        )
        response = TestClient(create_app(model_dir)).post(
            "/predict/text", json={"text": "godom po naszymu"}
        )
        assert response.status_code == 200
        header = response.headers["X-Model-Classes"]
        assert header.isascii()  # latin-1 safe
        assert {unquote(part) for part in header.split(",")} == {"śląsk", "kaszëby"}


class TestBatchPredict:
    def test_three_texts_return_three_predictions(self, client: TestClient) -> None:
        response = client.post("/predict/text/batch", json={"texts": [_PODHALE, _SILESIA, _KURPIE]})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload) == 3
        assert all("probabilities" in prediction for prediction in payload)
        assert response.headers["X-Model-Target"] == "dialect"

    def test_top_k_truncates_each_prediction(self, client: TestClient) -> None:
        response = client.post(
            "/predict/text/batch", json={"texts": [_PODHALE, _SILESIA], "top_k": 1}
        )
        assert response.status_code == 200
        assert all(len(prediction["probabilities"]) == 1 for prediction in response.json())

    def test_empty_list_is_rejected(self, client: TestClient) -> None:
        assert client.post("/predict/text/batch", json={"texts": []}).status_code == 400

    def test_blank_text_is_rejected(self, client: TestClient) -> None:
        assert (
            client.post("/predict/text/batch", json={"texts": [_PODHALE, "   "]}).status_code == 400
        )


class TestEnrichedHealth:
    def test_health_reports_class_count_and_abstention(self, client: TestClient) -> None:
        payload = client.get("/health").json()
        assert payload["n_classes"] == 3
        assert payload["abstain_threshold"] is None
        assert payload["abstain_enabled"] is False

    def test_health_does_not_disclose_the_server_filesystem_path(self, client: TestClient) -> None:
        """/health must not echo the local model path (CWE-200 info disclosure)."""
        payload = client.get("/health").json()
        assert "model" not in payload
        # No value should look like an absolute filesystem path.
        for value in payload.values():
            if isinstance(value, str):
                assert not value.startswith(("/", "C:\\", "\\\\"))


class TestOpenAPIAndDemo:
    def test_openapi_is_valid_json_listing_new_routes(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()  # raises if not valid JSON
        paths = spec["paths"]
        assert {"/", "/metrics", "/predict/text/batch"} <= set(paths)

    def test_demo_page_is_served_as_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "region-dot" in response.text


class TestDemoUnit:
    """The demo builder and projection are pure functions -- no server needed."""

    def test_demo_page_contains_expected_markup(self) -> None:
        from tulip.serve._demo import demo_page

        page = demo_page(title="My Tulip Demo")
        assert "My Tulip Demo" in page
        assert "classify-form" in page
        assert "<svg" in page
        assert 'data-region="podhale"' in page

    def test_projection_places_corners_at_canvas_edges(self) -> None:
        from tulip.labels.geo import POLAND_BOUNDS, GeoPoint
        from tulip.serve._demo import project

        south, west, north, east = POLAND_BOUNDS
        top_left = project(GeoPoint(north, west), POLAND_BOUNDS, width=200, height=100)
        bottom_right = project(GeoPoint(south, east), POLAND_BOUNDS, width=200, height=100)
        assert top_left == pytest.approx((0.0, 0.0))
        assert bottom_right == pytest.approx((200.0, 100.0))

    def test_projection_inverts_latitude(self) -> None:
        from tulip.labels.geo import POLAND_BOUNDS, GeoPoint
        from tulip.serve._demo import project

        northern = project(GeoPoint(53.0, 19.0), POLAND_BOUNDS, width=100, height=100)
        southern = project(GeoPoint(50.0, 19.0), POLAND_BOUNDS, width=100, height=100)
        assert northern[1] < southern[1]
