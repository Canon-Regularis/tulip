"""Deployment & MLOps: a content-addressed model registry above the artifacts.

The registry sits directly on the model-persistence layer, reusing its on-disk
format verbatim rather than re-implementing it; the registry logic itself adds
only the standard library and pydantic on top.
"""

from __future__ import annotations

from tulip.deploy.registry_store import (
    ModelRegistry,
    RegistryEntry,
    Stage,
    artifact_digest,
)

__all__ = [
    "ModelRegistry",
    "RegistryEntry",
    "Stage",
    "artifact_digest",
]
