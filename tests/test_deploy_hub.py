"""Tests for pushing registered models to the Hugging Face Hub (fully offline)."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest
from sklearn.linear_model import LogisticRegression
from typer.testing import CliRunner

from conftest import block_imports
from tulip.core.exceptions import DataError, MissingDependencyError
from tulip.deploy import ModelRegistry, hub_readme, push_to_hub
from tulip.models.persistence import save_model

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


class _FakeHfApi:
    """Records every Hub call so tests can assert on them without a network."""

    last: _FakeHfApi | None = None

    def __init__(self) -> None:
        self.created: dict[str, Any] | None = None
        self.uploaded_folder: dict[str, Any] | None = None
        self.uploaded_file: dict[str, Any] | None = None
        _FakeHfApi.last = self

    def create_repo(self, repo_id: str, **kwargs: Any) -> str:
        self.created = {"repo_id": repo_id, **kwargs}
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(self, **kwargs: Any) -> None:
        self.uploaded_folder = kwargs

    def upload_file(self, **kwargs: Any) -> None:
        self.uploaded_file = kwargs


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> type[_FakeHfApi]:
    module = types.SimpleNamespace(HfApi=_FakeHfApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return _FakeHfApi


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    model = LogisticRegression().fit([[0.0], [1.0]], ["a", "b"])
    model_dir = save_model(
        model, tmp_path / "model", metadata={"target": "dialect", "task": "text"}
    )
    store = ModelRegistry(tmp_path / "registry")
    store.add(model_dir, name="dia", version="1", metrics={"f1_macro": 0.912345})
    return store


def test_push_uploads_artifact_and_readme(
    registry: ModelRegistry, fake_hub: type[_FakeHfApi]
) -> None:
    url = push_to_hub(registry, "dia@1", repo_id="someone/tulip-dia")

    assert url == "https://huggingface.co/someone/tulip-dia"
    api = fake_hub.last
    assert api is not None
    assert api.created == {
        "repo_id": "someone/tulip-dia",
        "private": True,
        "exist_ok": True,
        "repo_type": "model",
    }
    entry = registry.resolve("dia@1")
    assert api.uploaded_folder is not None
    assert api.uploaded_folder["folder_path"] == str(registry.path_for(entry))
    assert api.uploaded_folder["repo_id"] == "someone/tulip-dia"
    assert entry.digest[:12] in api.uploaded_folder["commit_message"]
    assert api.uploaded_file is not None
    assert api.uploaded_file["path_in_repo"] == "README.md"
    readme = api.uploaded_file["path_or_fileobj"].decode("utf-8")
    assert "library_name: tulip" in readme
    assert entry.digest in readme
    assert "## Registry provenance" in readme
    assert "## Usage" in readme
    assert "f1_macro=0.9123" in readme
    # The usage snippet embeds the real repository id, and this artifact was
    # saved with save_model (not DialectClassifier.save), so it loads through
    # load_model.
    assert 'snapshot_download("someone/tulip-dia")' in readme
    assert "load_model" in readme
    assert "DialectClassifier.load" not in readme


def test_public_flag_creates_a_public_repo(
    registry: ModelRegistry, fake_hub: type[_FakeHfApi]
) -> None:
    push_to_hub(registry, "dia@1", repo_id="someone/tulip-dia", private=False)
    assert fake_hub.last is not None
    assert fake_hub.last.created is not None
    assert fake_hub.last.created["private"] is False


def test_readme_front_matter_carries_the_text_pipeline_tag(registry: ModelRegistry) -> None:
    entry = registry.resolve("dia@1")
    readme = hub_readme(entry, registry.path_for(entry), repo_id="someone/tulip-dia")
    assert readme.startswith("---\n")
    assert "pipeline_tag: text-classification" in readme
    assert "language:" in readme


def test_unknown_reference_raises_cleanly(
    registry: ModelRegistry, fake_hub: type[_FakeHfApi]
) -> None:
    with pytest.raises(DataError, match="no registered model"):
        push_to_hub(registry, "nope@9", repo_id="someone/x")


def test_missing_dependency_names_the_hf_extra(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    block_imports(monkeypatch, "huggingface_hub")
    # Match the install hint, not just "hf", which the module name also contains.
    with pytest.raises(MissingDependencyError, match=r"tulip-dialect\[hf\]"):
        push_to_hub(registry, "dia@1", repo_id="someone/x")


def test_cli_registry_push(
    registry: ModelRegistry, fake_hub: type[_FakeHfApi], tmp_path: Path
) -> None:
    from tulip.cli.app import app

    result = runner.invoke(
        app,
        [
            "registry",
            "push",
            "dia@1",
            "--repo-id",
            "someone/tulip-dia",
            "--registry",
            str(tmp_path / "registry"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "pushed dia@1" in result.output
