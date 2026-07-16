"""The serve command: HTTP inference service."""

from __future__ import annotations

from pathlib import Path

import typer

from tulip.cli._context import _tulip_errors, app


@app.command()
@_tulip_errors
def serve(
    model: Path = typer.Argument(
        ..., help="Saved model directory, or a registry reference (e.g. dialect@production)."
    ),
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Bind port."),
    registry: Path | None = typer.Option(
        None, "--registry", help="Registry root; then MODEL is a reference resolved from it."
    ),
) -> None:
    """Serve the model over HTTP (text + audio upload; needs the serve extra).

    Guards (auth, rate limit, concurrency, body-size cap, CORS, security headers)
    are read from ``TULIP_SERVE_*`` environment variables. With ``--registry`` the
    MODEL argument is a registry reference and the response carries
    ``X-Model-Version`` / ``X-Model-Digest``.
    """
    from tulip.deploy import ModelRegistry, artifact_digest
    from tulip.serve.app import create_app
    from tulip.utils.optional import optional_import

    if registry is not None:
        store = ModelRegistry(registry)
        entry = store.resolve(str(model))
        path, version, digest = store.path_for(entry), entry.version, entry.digest
    else:
        path, version, digest = model, None, artifact_digest(model)

    uvicorn = optional_import("uvicorn", extra="serve", purpose="the HTTP service")
    app_instance = create_app(path, model_version=version, model_digest=digest)
    uvicorn.run(app_instance, host=host, port=port)
