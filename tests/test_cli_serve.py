"""CLI smoke test for the serve command's dependency guard.

Actually binding a socket is out of scope for a unit test, and it would hang;
this only checks that ``serve`` fails cleanly when the serve extra is absent
rather than starting a server or raising an opaque traceback. When the extra is
installed the test is skipped, since the command would then try to bind a port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from tulip.cli.app import app
from tulip.utils import optional

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.mark.skipif(
    optional.is_available("fastapi") and optional.is_available("uvicorn"),
    reason="serve extra installed; invoking serve would bind a socket and block",
)
def test_serve_without_the_extra_fails_cleanly(tmp_path: Path) -> None:
    from conftest import make_manifest_experiment_config, write_manifest_corpus
    from tulip.data import DatasetBuilder
    from tulip.pipeline import DialectClassifier

    corpus = write_manifest_corpus(tmp_path / "corpus", speakers=6, variants=2)
    config = make_manifest_experiment_config(corpus, tmp_path / "artifacts", name="serve-cli")
    splits = DatasetBuilder(config.data).build(config.split, target=config.target)
    model_dir = tmp_path / "model"
    DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42).fit(
        splits.train
    ).save(model_dir)

    result = runner.invoke(app, ["serve", str(model_dir)])
    # No server started; the missing serve extra surfaces as a non-zero exit.
    assert result.exit_code != 0
