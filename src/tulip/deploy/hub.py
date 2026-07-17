"""Publish a registered model version to the Hugging Face Hub.

The local registry stays the source of truth. Pushing uploads the
content-addressed artifact directory unchanged (the model file plus its
``metadata.json`` sidecar) and adds a ``README.md`` rendered from the model
card, so the Hub page documents what the model is and how to load it.

Credentials are resolved by the ``huggingface_hub`` library itself, from the
cached login or the ``HF_TOKEN`` environment variable. This module never reads,
stores, or logs a token.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tulip.utils.logging import get_logger
from tulip.utils.optional import optional_import

if TYPE_CHECKING:
    from pathlib import Path

    from tulip.deploy.registry_store import ModelRegistry, RegistryEntry

__all__ = ["hub_readme", "push_to_hub"]

_logger = get_logger(__name__)


def push_to_hub(
    registry: ModelRegistry, reference: str, *, repo_id: str, private: bool = True
) -> str:
    """Resolve ``reference`` in the registry and upload it as a Hub model repo.

    Args:
        registry: The local model registry to resolve against.
        reference: A registry reference (``name@version`` or ``name@stage``).
        repo_id: Target repository, for example ``someone/tulip-dialect``.
        private: Visibility used when the repository is first created; an
            existing repository keeps its current visibility.

    Returns:
        The repository URL.

    Raises:
        MissingDependencyError: if ``huggingface_hub`` is not installed
            (extra ``hf``).
        DataError: if the reference does not resolve or the artifact sidecar
            is unreadable.
    """
    hub = optional_import(
        "huggingface_hub", extra="hf", purpose="pushing models to the Hugging Face Hub"
    )
    entry = registry.resolve(reference)
    artifact_dir = registry.path_for(entry)
    readme = hub_readme(entry, artifact_dir, repo_id=repo_id)

    api = hub.HfApi()
    url = str(api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model"))
    api.upload_folder(
        folder_path=str(artifact_dir),
        repo_id=repo_id,
        commit_message=f"tulip {entry.name}@{entry.version} ({entry.digest[:12]})",
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message=f"model card for {entry.name}@{entry.version}",
    )
    _logger.info("pushed %s@%s to %s", entry.name, entry.version, url)
    return url


def hub_readme(entry: RegistryEntry, artifact_dir: Path, *, repo_id: str) -> str:
    """Render the Hub ``README.md`` for one registry entry.

    A YAML front matter block carries the Hub metadata, followed by the same
    model card the ``tulip card model`` command renders, a registry section
    (version, stage, digest, recorded metrics), and a load snippet for the
    given repository id.
    """
    from tulip._jsonio import read_json_object
    from tulip.evaluation.cards import model_card
    from tulip.models.persistence import METADATA_FILENAME

    sidecar_path = artifact_dir / METADATA_FILENAME
    sidecar = read_json_object(sidecar_path, what="artifact sidecar")

    stored = sidecar.get("metadata", {})
    is_dialect_classifier = isinstance(stored, dict) and stored.get("kind") == "DialectClassifier"
    front_matter = _front_matter(entry)
    card = model_card(sidecar, {})
    registry_section = _registry_section(entry)
    usage = _usage_section(repo_id, dialect_classifier=is_dialect_classifier)
    return "\n".join([front_matter, card, registry_section, usage])


def _front_matter(entry: RegistryEntry) -> str:
    """The Hub metadata block. Only fields with known values are emitted."""
    lines = [
        "---",
        "library_name: tulip",
        "language:",
        "- pl",
        "tags:",
        "- dialect-identification",
        "- polish",
    ]
    if entry.task == "text":
        lines.append("pipeline_tag: text-classification")
    elif entry.task == "audio":
        lines.append("pipeline_tag: audio-classification")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _registry_section(entry: RegistryEntry) -> str:
    """The provenance block tying the Hub repo back to the local registry."""
    lines = [
        "",
        "## Registry provenance",
        "",
        f"- Name: `{entry.name}`",
        f"- Version: `{entry.version}`",
        f"- Stage: `{entry.stage.value}`",
        f"- Content digest: `{entry.digest}`",
        f"- tulip version: `{entry.tulip_version}`",
    ]
    if entry.metrics:
        lines.append(
            "- Recorded metrics: "
            + ", ".join(f"`{key}={value:.4f}`" for key, value in sorted(entry.metrics.items()))
        )
    return "\n".join(lines) + "\n"


def _usage_section(repo_id: str, *, dialect_classifier: bool) -> str:
    """How to load the pushed model back into tulip.

    Artifacts saved by ``DialectClassifier.save`` load through the classifier
    facade; anything else saved with ``save_model`` loads through
    ``load_model``, which the snippet must reflect or it would raise.
    """
    if dialect_classifier:
        return (
            "\n## Usage\n\n"
            "```python\n"
            "from huggingface_hub import snapshot_download\n"
            "from tulip.pipeline import DialectClassifier\n\n"
            f'path = snapshot_download("{repo_id}")\n'
            "classifier = DialectClassifier.load(path)\n"
            'print(classifier.predict("baca pognal owce na hale"))\n'
            "```\n"
        )
    return (
        "\n## Usage\n\n"
        "```python\n"
        "from huggingface_hub import snapshot_download\n"
        "from tulip.models import load_model\n\n"
        f'path = snapshot_download("{repo_id}")\n'
        "model, sidecar = load_model(path)\n"
        "```\n"
    )
