"""The shipped examples stay runnable end to end on the core install.

They execute the real train/evaluate/predict path on the synthetic corpus, so a
break in the public API (a renamed helper, a changed signature) fails here
instead of in a user's first five minutes.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("script", ["train_and_predict.py", "compare_models.py"])
def test_example_script_runs(script: str, capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(_EXAMPLES / script), run_name="__main__")
    assert capsys.readouterr().out.strip()  # it produced output


def test_tutorial_notebook_code_matches_the_script() -> None:
    # The notebook mirrors train_and_predict.py; keep its code cells free of
    # imports the script does not use, so the "runs offline" promise holds.
    import json

    notebook = json.loads((_EXAMPLES / "tutorial.ipynb").read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "generate_corpus" in code
    assert "DialectClassifier" in code
    assert "evaluate_samples" in code
