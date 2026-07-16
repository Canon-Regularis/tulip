"""The analyze command: selective prediction and error analysis over a dump."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from tulip.cli._context import _console, _tulip_errors, app


@app.command()
@_tulip_errors
def analyze(
    predictions_path: Path = typer.Argument(
        ..., help="A predictions_<split>.json written by `tulip train`."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the reports as JSON."),
    top_k: int = typer.Option(10, "--top-k", min=1, help="Confused pairs / exemplars to show."),
    hierarchical: bool = typer.Option(
        False, "--hierarchical", help="Also report family/dialect hierarchical metrics."
    ),
    power: bool = typer.Option(
        False, "--power", help="Also report the minimum detectable effect at this sample size."
    ),
    fairness: bool = typer.Option(
        False, "--fairness", help="Also report subgroup disparity across slices."
    ),
) -> None:
    """Analyse a saved per-sample predictions dump: selective prediction + errors.

    Turns the ``predictions_test.json`` a training run leaves behind into an
    operator's diagnosis: a risk-coverage curve (accuracy at each coverage, AURC)
    and an error report (most-confused pairs, hardest mistakes, and per-slice
    fairness metrics). ``--hierarchical`` adds partial credit for a family-correct
    dialect miss; ``--power`` reports the smallest detectable accuracy gap;
    ``--fairness`` reports the worst-versus-best subgroup gap per slice.
    """
    from tulip.evaluation.error_analysis import error_report
    from tulip.evaluation.fairness import fairness_report
    from tulip.evaluation.hierarchical_metrics import hierarchical_metrics
    from tulip.evaluation.power import minimum_detectable_effect
    from tulip.evaluation.predictions import SplitPredictions
    from tulip.evaluation.selective import selective_report

    predictions = SplitPredictions.load(predictions_path)
    selective = selective_report(predictions)
    errors = error_report(predictions, top_k=top_k)
    hier = (
        hierarchical_metrics(predictions.true_labels(), predictions.pred_labels())
        if hierarchical
        else None
    )
    mde = minimum_detectable_effect(len(predictions)) if power else None
    fair = fairness_report(predictions) if fairness else None

    if json_output:
        payload: dict[str, Any] = {
            "selective": selective.model_dump(),
            "errors": errors.model_dump(),
        }
        if hier is not None:
            payload["hierarchical"] = hier.model_dump()
        if mde is not None:
            payload["power"] = mde.model_dump()
        if fair is not None:
            payload["fairness"] = fair.model_dump()
        _console.print_json(data=payload)
        return
    _console.print(selective.to_markdown())
    _console.print()
    _console.print(errors.to_markdown())
    for extra in (hier, mde, fair):
        if extra is not None:
            _console.print()
            _console.print(extra.to_markdown())
