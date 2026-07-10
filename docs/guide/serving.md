# Serving

tulip ships a FastAPI service that wraps one saved model behind a single HTTP
interface for typed text and uploaded audio. It is built by
`tulip.serve.create_app(model_path)` and exposed on the CLI as `tulip serve`.

!!! note "Optional dependency"
    The service needs the `serve` extra (FastAPI + uvicorn):
    `pip install -e ".[serve]"`. These imports are lazy, so `import tulip.serve`
    never requires them.

## Starting the service

```bash
tulip serve artifacts/synthetic-text/model
```

This loads the saved [`DialectClassifier`](../reference/pipeline.md) once at
startup and serves it. The model's task (text vs audio) and its class list are
fixed by the artifact; endpoints that do not match the model's modality return a
`400` rather than guessing.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Browser demo UI — paste text and see the prediction. |
| `GET` | `/health` | Liveness plus model identity (version, task, target, classes). |
| `POST` | `/predict/text` | Classify one text: JSON body `{"text": ..., "top_k": ...}`. |
| `POST` | `/predict/text/batch` | Classify many texts in one request. |
| `POST` | `/predict/audio` | Classify an uploaded audio file (multipart, audio models). |
| `GET` | `/metrics` | Prometheus exposition of request/latency counters. |

Responses are [`Prediction`](../reference/pipeline.md) JSON — the label, the full
probability distribution, top-k, and the abstention flag — serialised natively by
pydantic.

### Text

```bash
curl -X POST localhost:8000/predict/text \
  -H "content-type: application/json" \
  -d '{"text": "Jo żech je z Katowic i godom po naszymu.", "top_k": 3}'
```

`top_k` is optional and truncates the returned distribution. Blank text is a
`400`. A text request against an audio-only model is a `400` with a message
telling you to train with `task: text`.

### Batch text

The batch endpoint takes a list of texts and returns a list of predictions in the
same order, so a client can amortise HTTP overhead over many short utterances.

### Audio

```bash
curl -X POST "localhost:8000/predict/audio?format=wav" \
  -F "file=@clip.wav"
```

The upload suffix is sanity-checked against the accepted container formats
(`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.opus`); decoding happens later in the
feature extractor. An undecodable upload is treated as a bad request (`400`), not
a server fault.

## Observability

- **Request IDs** — each request is tagged with an identifier that flows through
  the logs and the response, so a prediction can be traced back to its call.
- **`/metrics`** — a Prometheus endpoint exposing request counts and latency, so
  the service drops straight into a standard scrape-and-dashboard setup.

## Programmatic use

`create_app` returns a plain `fastapi.FastAPI` instance, so you can mount it in a
larger application or drive it with `TestClient`:

```python
from tulip.serve import create_app

app = create_app("artifacts/synthetic-text/model")
```

See [`tulip.serve`](../reference/index.md) for the request models and endpoint
contract.
