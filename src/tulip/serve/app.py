"""HTTP inference service: one interface for typed text and uploaded audio.

``create_app`` loads a saved :class:`~tulip.pipeline.DialectClassifier` once and
exposes it over an observable, self-documenting API:

* ``GET /``: a self-contained demo web page (see :mod:`tulip.serve._demo`).
* ``GET /health``: liveness plus model identity and abstention config.
* ``GET /metrics``: Prometheus text exposition (see :mod:`tulip.serve._metrics`).
* ``POST /predict/text``: JSON body ``{"text": ..., "top_k": ...}``.
* ``POST /predict/text/batch``: JSON body ``{"texts": [...], "top_k": ...}``.
* ``POST /predict/audio``: multipart file upload (audio-trained models).

Every request flows through one ASGI middleware that assigns a correlation ID,
times the handler, stamps ``X-Request-ID`` / ``X-Process-Time-Ms`` response
headers, records Prometheus metrics, and emits one structured log line.

Prediction responses are :class:`~tulip.core.types.Prediction` JSON, enriched
with ``X-Tulip-Version`` / ``X-Model-Target`` / ``X-Model-Classes`` headers so a
client learns the model identity without a second call. FastAPI and uvicorn are
optional (extra ``serve``); this module imports them lazily so ``import
tulip.serve`` never requires them.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from tulip import __version__
from tulip.core.exceptions import DataError
from tulip.core.types import Prediction, TaskType
from tulip.serve._demo import demo_page
from tulip.serve._guards import install_guards
from tulip.serve._metrics import CONTENT_TYPE, MetricsRegistry
from tulip.serve.settings import MAX_BATCH_CEILING, ServeSettings
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    # Runtime-only for FastAPI: these names are imported lazily (fastapi is an
    # optional extra), so they live here for the type checker. ``Response`` is
    # additionally injected into module globals inside ``create_app`` because
    # FastAPI resolves endpoint annotations against them at route registration.
    from fastapi import Request, Response

_logger = get_logger(__name__)

#: Upload suffixes accepted by the audio endpoint (decoding happens later,
#: in the feature extractor; this is just a first-line sanity filter).
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}

#: Concrete dialect sentences reused across OpenAPI examples so ``/docs`` shows
#: real Polish rather than "string".
_PODHALE_EXAMPLE = "Hej, baca się pyto, kaj się owce pasą na holi."
_SILESIA_EXAMPLE = "Jo żech je z Katowic i godom po naszymu cołki czos."


class TextRequest(BaseModel):
    """Request body for ``POST /predict/text``."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"text": _PODHALE_EXAMPLE, "top_k": 3}]}
    )

    text: str = Field(
        min_length=1, description="The text to classify.", examples=[_PODHALE_EXAMPLE]
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Truncate the returned distribution to k classes.",
        examples=[3],
    )


class BatchTextRequest(BaseModel):
    """Request body for ``POST /predict/text/batch``.

    ``texts`` is capped at :data:`~tulip.serve.settings.MAX_BATCH_CEILING` at the
    schema level (an oversized list is a 422); the configurable ``max_batch`` can
    only be lower, and the handler enforces it. Emptiness and per-item blankness
    are enforced in the handler
    so they surface as a 400, the same "your content is unusable" semantics as
    the single-text endpoint, rather than a schema-shaped 422.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"texts": [_PODHALE_EXAMPLE, _SILESIA_EXAMPLE], "top_k": None}]
        }
    )

    texts: list[str] = Field(
        max_length=MAX_BATCH_CEILING,
        description=f"One to {MAX_BATCH_CEILING} texts to classify in a single call.",
        examples=[[_PODHALE_EXAMPLE, _SILESIA_EXAMPLE]],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Truncate every returned distribution to k classes.",
        examples=[None],
    )


def _truncated(prediction: Prediction, top_k: int | None) -> Prediction:
    """Return a copy limited to the top-k classes (full distribution otherwise)."""
    if top_k is None or top_k >= len(prediction.probabilities):
        return prediction
    return prediction.model_copy(update={"probabilities": prediction.top_k(top_k)})


#: HTTP methods labelled verbatim in metrics. Anything else (an arbitrary
#: extension method a client can invent) collapses to ``OTHER`` so the method
#: label cannot explode metric cardinality.
_KNOWN_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})

#: Metric ``path`` label for a request that matched no route (a 404). Using a
#: constant instead of the raw URL is what keeps cardinality bounded: otherwise
#: an unauthenticated client hitting ``/aaa``, ``/bbb``, ... would add a
#: permanent metric series per distinct path and grow memory without bound.
_UNMATCHED_PATH = "<unmatched>"


def _path_template(request: Request) -> str:
    """Return the matched route *template* for metrics, not the concrete URL.

    Using the template (``/predict/text``) rather than ``request.url.path`` keeps
    metric cardinality bounded even if a future route grows path parameters. A
    request that matched no route (a 404) is labelled with a single
    :data:`_UNMATCHED_PATH` bucket rather than its attacker-controlled raw path.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    return template if isinstance(template, str) else _UNMATCHED_PATH


def _metric_method(method: str) -> str:
    """Normalise an HTTP method to the bounded :data:`_KNOWN_METHODS` set."""
    return method if method in _KNOWN_METHODS else "OTHER"


def create_app(
    model_path: Path | str,
    *,
    model_version: str | None = None,
    model_digest: str | None = None,
    settings: ServeSettings | None = None,
) -> Any:
    """Build the FastAPI application around one saved model artifact.

    Guards (auth, rate limit, concurrency cap, request-body ceiling, CORS,
    security headers) are configured from ``TULIP_SERVE_*`` environment variables
    (see :class:`~tulip.serve.settings.ServeSettings`) and installed *inside* the
    observability middleware, so a guard-rejected request is still timed, counted,
    and logged.

    Args:
        model_path: Directory written by :meth:`DialectClassifier.save`.
        model_version: Registry version of the model, surfaced as
            ``X-Model-Version`` when set (the serving layer resolves it).
        model_digest: Content digest of the artifact, surfaced as
            ``X-Model-Digest`` when set.
        settings: Guard settings; defaults to
            :meth:`ServeSettings.from_env`. Pass explicitly to configure the
            guards in-process (e.g. in tests).

    Returns:
        A configured :class:`fastapi.FastAPI` instance.

    Raises:
        MissingDependencyError: if FastAPI is not installed (extra ``serve``).
        DataError: if the model artifact is missing or corrupt.
    """
    fastapi = optional_import("fastapi", extra="serve", purpose="the HTTP service")
    from tulip.pipeline import DialectClassifier  # deferred: heavy sklearn import chain

    # FastAPI resolves endpoint annotations (``response: Response``) against this
    # module's globals at route-registration time. ``Response`` cannot be a
    # top-level import (fastapi is optional), so publish it now, before any
    # endpoint below is defined, rather than annotating with the lazily
    # imported type, which the NOTE on the audio endpoint forbids.
    globals()["Response"] = fastapi.Response
    # Used only inside the middleware closure (not an endpoint annotation), so a
    # local binding of the lazily-imported type is enough.
    json_response = fastapi.responses.JSONResponse

    classifier = DialectClassifier.load(model_path)
    _logger.info(
        "serving %s (task=%s, %d classes)",
        model_path,
        classifier.task.value,
        len(classifier.classes_),
    )

    app = fastapi.FastAPI(
        title="tulip",
        description="Polish dialect detection service",
        version=__version__,
    )
    # Endpoints close over `classifier` directly; only the path is state.
    app.state.model_path = str(model_path)
    registry = MetricsRegistry()
    app.state.metrics = registry

    settings = settings if settings is not None else ServeSettings.from_env()
    app.state.serve_settings = settings
    # Install guards BEFORE the observability middleware below, so that
    # observability (added last) stays the outermost layer and still times,
    # counts, and logs any guard-rejected request. See serve._guards.
    install_guards(app, settings)

    def _set_identity_headers(response: Response) -> None:
        """Stamp model-identity headers so a client sees the model without /health.

        HTTP header values are latin-1 only, but dialect class labels are Polish
        and routinely carry diacritics (``śląsk``, ``kaszëby``). A raw
        ``",".join`` would raise ``UnicodeEncodeError`` and 500 *every* response.
        Each label is percent-encoded (RFC 3986), which is ASCII-safe, keeps
        plain ASCII labels readable, and makes the comma an unambiguous separator
        even when a label itself contains a comma. Clients unquote per field.

        ``X-Model-Version`` / ``X-Model-Digest`` identify the exact registered
        artifact when the model was resolved from the registry (both omitted
        otherwise).
        """
        response.headers["X-Tulip-Version"] = __version__
        response.headers["X-Model-Target"] = classifier.target.value
        response.headers["X-Model-Classes"] = ",".join(
            quote(label, safe="") for label in classifier.classes_
        )
        if model_version is not None:
            response.headers["X-Model-Version"] = quote(model_version, safe="")
        if model_digest is not None:
            response.headers["X-Model-Digest"] = model_digest

    @app.middleware("http")
    async def observability(request: Request, call_next: Any) -> Any:
        """Assign a correlation ID, time the handler, record metrics, and log once.

        A handler that raises an *unhandled* exception (a genuine 500; handled
        HTTPExceptions are already turned into responses inside ``call_next``)
        must not slip past observability: it is still timed, counted as a 500,
        logged with its traceback and correlation ID, and answered with a clean
        JSON 500 that still carries the ``X-Request-ID``/``X-Process-Time-Ms``
        headers, rather than an unheaded, uncounted, unlogged failure.
        """
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000.0
            _record(request, request_id, status=500, duration_ms=duration_ms, level="exception")
            return json_response(
                status_code=500,
                content={"detail": "internal server error"},
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time-Ms": f"{duration_ms:.3f}",
                },
            )
        duration_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.3f}"
        _record(request, request_id, status=response.status_code, duration_ms=duration_ms)
        return response

    def _record(
        request: Request,
        request_id: str,
        *,
        status: int,
        duration_ms: float,
        level: str = "info",
    ) -> None:
        """Record one request into metrics and the structured log."""
        method = _metric_method(request.method)
        path = _path_template(request)
        registry.observe(method=method, path=path, status=status, duration_ms=duration_ms)
        log = _logger.exception if level == "exception" else _logger.info
        log(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.3f",
            request_id,
            method,
            path,
            status,
            duration_ms,
        )

    @app.get(
        "/",
        response_class=fastapi.responses.HTMLResponse,
        tags=["demo"],
        summary="Interactive demo UI",
        description="A self-contained HTML page that classifies text and maps the result.",
    )
    def demo() -> str:
        return demo_page(title="tulip: Polish dialect detection")

    @app.get(
        "/health",
        tags=["operations"],
        summary="Liveness and model identity",
        description="Report service status plus the loaded model's task, classes, and "
        "abstention configuration.",
    )
    def health() -> dict[str, Any]:
        # The server-side model path is NOT returned: on a non-loopback bind it
        # would disclose the OS account, directory layout, and any confidential
        # codenames in the path to any unauthenticated client (CWE-200). Model
        # identity is conveyed by version/task/target/classes; the path is kept
        # to server-side logs only.
        return {
            "status": "ok",
            "version": __version__,
            "model_version": model_version,
            "model_digest": model_digest,
            "task": classifier.task.value,
            "target": classifier.target.value,
            "classes": list(classifier.classes_),
            "n_classes": len(classifier.classes_),
            "abstain_threshold": classifier.abstain_threshold,
            "abstain_enabled": classifier.abstain_threshold is not None,
        }

    @app.get(
        "/metrics",
        tags=["operations"],
        summary="Prometheus metrics",
        description="Request counts and per-path latency in Prometheus text exposition format.",
    )
    def metrics() -> Response:
        return fastapi.Response(content=registry.render(), media_type=CONTENT_TYPE)

    def _require_text_task() -> None:
        """Reject a non-text model with the 400 the text endpoints share."""
        if classifier.task is not TaskType.TEXT:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"this model classifies {classifier.task.value}, not text",
            )

    @app.post(
        "/predict/text",
        response_model=Prediction,
        tags=["inference"],
        summary="Classify one Polish text",
        description="Return the full ranked dialect probability distribution for one text "
        "(text-trained models only).",
    )
    def predict_text(request: TextRequest, response: Response) -> Prediction:
        _require_text_task()
        if not request.text.strip():
            raise fastapi.HTTPException(status_code=400, detail="text must not be blank")
        result = _truncated(classifier.predict(request.text), request.top_k)
        _set_identity_headers(response)
        return result

    @app.post(
        "/predict/text/batch",
        response_model=list[Prediction],
        tags=["inference"],
        summary="Classify a batch of Polish texts",
        description="Classify up to "
        f"{MAX_BATCH_CEILING} texts in one call, returning one prediction per input in order.",
    )
    def predict_text_batch(request: BatchTextRequest, response: Response) -> list[Prediction]:
        _require_text_task()
        if not request.texts:
            raise fastapi.HTTPException(status_code=400, detail="texts must not be empty")
        if len(request.texts) > settings.max_batch:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"at most {settings.max_batch} texts per batch (configured limit)",
            )
        if any(not text.strip() for text in request.texts):
            raise fastapi.HTTPException(status_code=400, detail="every text must be non-blank")
        predictions = classifier.predict_batch(list(request.texts))
        _set_identity_headers(response)
        return [_truncated(prediction, request.top_k) for prediction in predictions]

    # NOTE: endpoint annotations must resolve from module globals (FastAPI
    # calls get_type_hints under `from __future__ import annotations`), so
    # only builtins and module-level names appear in the signatures below,
    # never the lazily imported fastapi types. `Response` satisfies this via
    # the globals() injection at the top of create_app.
    @app.post(
        "/predict/audio",
        response_model=Prediction,
        tags=["inference"],
        summary="Classify an uploaded audio clip",
        description="Classify a multipart audio upload (audio-trained models only).",
    )
    def predict_audio(
        response: Response,
        file: bytes = fastapi.File(..., description="The audio file content."),
        audio_format: str = fastapi.Query(
            "wav", alias="format", description="Audio container format of the upload."
        ),
        top_k: int | None = fastapi.Query(default=None, ge=1),
    ) -> Prediction:
        if classifier.task is not TaskType.AUDIO:
            raise fastapi.HTTPException(
                status_code=400,
                detail=(
                    f"this model classifies {classifier.task.value}, not audio; "
                    "train with task: audio to enable this endpoint"
                ),
            )
        suffix = f".{audio_format.strip().lstrip('.').lower()}"
        if suffix not in _AUDIO_SUFFIXES:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"unsupported audio format {suffix!r}; "
                f"expected one of {sorted(_AUDIO_SUFFIXES)}",
            )
        if not file:
            raise fastapi.HTTPException(status_code=400, detail="uploaded file is empty")
        # NamedTemporaryFile must be closed before reopening on Windows, hence
        # delete=False plus explicit cleanup.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(file)
            temp_path = Path(handle.name)
        try:
            prediction = classifier.predict(temp_path)
        except DataError as exc:
            # The suffix is validated above, but the *bytes* are not: a caller can
            # upload anything under a .wav name. An undecodable upload is a bad
            # request, not a server fault, so do not let it surface as a 500.
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"uploaded file could not be decoded as {suffix} audio",
            ) from exc
        finally:
            temp_path.unlink(missing_ok=True)
        _set_identity_headers(response)
        return _truncated(prediction, top_k)

    return app


__all__ = ["BatchTextRequest", "TextRequest", "create_app"]
