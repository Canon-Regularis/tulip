# Deployment and serving

The model registry and the serving guard configuration. See the
[Serving guide](../guide/serving.md) for the walk-through.

## Model registry

A content-addressed store over the persisted model artifacts. It adds versioned
entries, SHA-256 integrity, and a staging-to-production promotion flow with
one-command rollback.

::: tulip.deploy.ModelRegistry

::: tulip.deploy.RegistryEntry

::: tulip.deploy.Stage

::: tulip.deploy.artifact_digest

## Serving settings

Environment-driven configuration for the serving guards: auth, rate limit,
concurrency cap, request-body ceiling, CORS, and security headers.

::: tulip.serve.settings.ServeSettings
