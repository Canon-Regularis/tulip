"""Tests for the `tulip registry` CLI command group."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sklearn.linear_model import LogisticRegression
from typer.testing import CliRunner

from tulip.cli.app import app
from tulip.models.persistence import save_model

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    model = LogisticRegression().fit([[0.0], [1.0]], ["a", "b"])
    return save_model(model, tmp_path / "model", metadata={"target": "dialect", "task": "text"})


def _run(args: list[str]) -> str:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result.output


class TestRegistryCli:
    def test_add_promote_resolve_flow(self, model_dir: Path, tmp_path: Path) -> None:
        root = str(tmp_path / "reg")
        add = _run(
            [
                "registry",
                "add",
                str(model_dir),
                "--name",
                "dia",
                "--version",
                "1",
                "--registry",
                root,
            ]
        )
        assert "registered dia@1" in add

        _run(["registry", "promote", "dia", "1", "--registry", root])
        resolved = _run(["registry", "resolve", "dia", "--registry", root])
        assert "dia@1 (production)" in resolved
        assert "digest:" in resolved

    def test_ls_shows_the_entry(self, model_dir: Path, tmp_path: Path) -> None:
        root = str(tmp_path / "reg")
        _run(
            [
                "registry",
                "add",
                str(model_dir),
                "--name",
                "dia",
                "--version",
                "1",
                "--registry",
                root,
            ]
        )
        output = _run(["registry", "ls", "--registry", root])
        assert "dia" in output and "staging" in output

    def test_rollback(self, model_dir: Path, tmp_path: Path) -> None:
        root = str(tmp_path / "reg")
        for version in ("1", "2"):
            _run(
                [
                    "registry",
                    "add",
                    str(model_dir),
                    "--name",
                    "dia",
                    "--version",
                    version,
                    "--registry",
                    root,
                ]
            )
            _run(["registry", "promote", "dia", version, "--registry", root])
        rolled = _run(["registry", "rollback", "dia", "--registry", root])
        assert "dia@1 is now production" in rolled

    def test_unknown_reference_is_a_clean_error(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["registry", "resolve", "ghost", "--registry", str(tmp_path)])
        assert result.exit_code == 1
        assert "error:" in result.output
