"""HTTP inference service: one interface for typed text and uploaded audio.

``create_app`` loads a saved :class:`~tulip.pipeline.DialectClassifier` once
and exposes it over three endpoints:

* ``GET /health`` -- liveness plus model identity.
* ``POST /predict/text`` -- JSON body ``{"text": ..., "top_k": ...}``.
* ``POST /predict/audio`` -- multipart file upload (audio-trained models).

Responses are :class:`~tulip.core.types.Prediction` JSON. FastAPI and
uvicorn are optional (extra ``serve``); this module imports them lazily so
``import tulip.serve`` never requires them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tulip import __version__
from tulip.core.exceptions import DataError
from tulip.core.types import Prediction, TaskType
from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

_logger = get_logger(__name__)

#: Upload suffixes accepted by the audio endpoint (decoding happens later,
#: in the feature extractor; this is just a first-line sanity filter).
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


class TextRequest(BaseModel):
    """Request body for ``POST /predict/text``."""

    text: str = Field(min_length=1, description="The text to classify.")
    top_k: int | None = Field(
        default=None, ge=1, description="Truncate the returned distribution to k classes."
    )


def _truncated(prediction: Prediction, top_k: int | None) -> Prediction:
    """Return a copy limited to the top-k classes (full distribution otherwise)."""
    if top_k is None or top_k >= len(prediction.probabilities):
        return prediction
    return prediction.model_copy(update={"probabilities": prediction.top_k(top_k)})


def create_app(model_path: Path | str) -> Any:
    """Build the FastAPI application around one saved model artifact.

    Args:
        model_path: Directory written by :meth:`DialectClassifier.save`.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.

    Raises:
        MissingDependencyError: if FastAPI is not installed (extra ``serve``).
        DataError: if the model artifact is missing or corrupt.
    """
    fastapi = optional_import("fastapi", extra="serve", purpose="the HTTP service")
    from tulip.pipeline import DialectClassifier  # deferred: heavy sklearn import chain

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

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "model": app.state.model_path,
            "task": classifier.task.value,
            "target": classifier.target.value,
            "classes": list(classifier.classes_),
        }

    @app.post("/predict/text", response_model=Prediction)
    def predict_text(request: TextRequest) -> Prediction:
        if classifier.task is not TaskType.TEXT:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"this model classifies {classifier.task.value}, not text",
            )
        if not request.text.strip():
            raise fastapi.HTTPException(status_code=400, detail="text must not be blank")
        return _truncated(classifier.predict(request.text), request.top_k)

    # NOTE: endpoint annotations must resolve from module globals (FastAPI
    # calls get_type_hints under `from __future__ import annotations`), so
    # only builtins and module-level names appear in the signatures below --
    # never the lazily imported fastapi types.
    @app.post("/predict/audio", response_model=Prediction)
    def predict_audio(
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
            return _truncated(classifier.predict(temp_path), top_k)
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

    return app


__all__ = ["TextRequest", "create_app"]
