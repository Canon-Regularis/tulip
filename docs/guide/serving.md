# Serving

tulip ships a FastAPI service. It wraps one saved model behind a single HTTP
interface for typed text and uploaded audio. Build it with
`tulip.serve.create_app(model_path)`. On the CLI it is `tulip serve`.

!!! note "Optional dependency"
    The service needs the `serve` extra (FastAPI and uvicorn):
    `pip install -e ".[serve]"`. These imports are lazy, so `import tulip.serve`
    never requires them.

## Starting the service

```bash
tulip serve artifacts/synthetic-text/model
```

This loads the saved [`DialectClassifier`](../reference/pipeline.md) once at
startup. The model's task (text or audio) and its class list come from the
artifact. An endpoint that does not match the model's modality returns a `400`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Browser demo UI. Paste text and see the prediction. |
| `GET` | `/health` | Liveness plus model identity. |
| `POST` | `/predict/text` | Classify one text. |
| `POST` | `/predict/text/batch` | Classify many texts in one request. |
| `POST` | `/predict/audio` | Classify an uploaded audio file. |
| `GET` | `/metrics` | Prometheus request and latency counters. |

Responses are [`Prediction`](../reference/pipeline.md) JSON: the label, the full
probability distribution, top-k, and the abstention flag.

### Text

```bash
curl -X POST localhost:8000/predict/text \
  -H "content-type: application/json" \
  -d '{"text": "Jo żech je z Katowic i godom po naszymu.", "top_k": 3}'
```

`top_k` is optional and truncates the distribution. Blank text is a `400`. A text
request against an audio-only model is a `400`.

### Batch text

The batch endpoint takes a list of texts. It returns a list of predictions in the
same order. This amortises HTTP overhead over many short utterances.

### Audio

```bash
curl -X POST "localhost:8000/predict/audio?format=wav" \
  -F "file=@clip.wav"
```

The upload suffix is checked against the accepted formats (`.wav`, `.mp3`,
`.flac`, `.ogg`, `.m4a`, `.opus`). Decoding happens later. An undecodable upload
is a `400`, not a server error.

## Observability

- **Request IDs.** Each request gets an ID. It flows through the logs and the
  response, so you can trace a prediction back to its call.
- **`/metrics`.** A Prometheus endpoint with request counts and latency. It drops
  into a standard scrape-and-dashboard setup.

## Guards

The service is safe on loopback out of the box. Before it faces a network, enable
the guards through `TULIP_SERVE_*` environment variables. Most are off by
default. The body-size ceiling and security headers are on. Guards run inside the
observability layer, so a rejected request is still timed, counted, and logged.

| Variable | Effect | Default |
| --- | --- | --- |
| `TULIP_SERVE_API_TOKEN` | Require `Authorization: Bearer <token>` (except `/health`, `/metrics`). | off |
| `TULIP_SERVE_RATE_LIMIT` | Per-client requests per minute. | off |
| `TULIP_SERVE_MAX_CONCURRENCY` | Maximum in-flight requests (`503` when full). | off |
| `TULIP_SERVE_MAX_BODY_BYTES` | Request-body ceiling, enforced before buffering (`413`). | 32 MiB |
| `TULIP_SERVE_MAX_BATCH` | Maximum texts per batch call. | 512 |
| `TULIP_SERVE_CORS_ORIGINS` | Comma-separated allowed CORS origins. | off |
| `TULIP_SERVE_SECURITY_HEADERS` | Add `X-Content-Type-Options`, CSP, and more. | on |
| `TULIP_SERVE_HSTS` | Add `Strict-Transport-Security` (HTTPS only). | off |

The body-size guard is the important one. `POST /predict/audio` reads the whole
upload into memory. The ceiling rejects an oversized upload with a `413` before
any of it is buffered.

## Model registry and versioning

For anything beyond a single model directory, register artifacts in a
content-addressed [model registry](../reference/deploy.md). Each `(name, version)`
gets a SHA-256 digest and a lifecycle stage. One command promotes or rolls back
the production model.

```bash
tulip registry add artifacts/run/model --name dialect --version 1 \
  --report artifacts/run/report_test.json
tulip registry promote dialect 1            # to production; archives the previous one
tulip registry rollback dialect             # one-command rollback
tulip serve dialect@production --registry artifacts/registry
```

Serving from the registry stamps `X-Model-Version` and `X-Model-Digest` on every
prediction. A client or an audit can then tell which artifact answered.

## Programmatic use

`create_app` returns a plain `fastapi.FastAPI` instance. You can mount it in a
larger app or drive it with `TestClient`:

```python
from tulip.serve import create_app

app = create_app("artifacts/synthetic-text/model")
```

See [`tulip.serve`](../reference/index.md) for the request models and the
endpoint contract.
