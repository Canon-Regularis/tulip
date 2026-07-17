"""Single-sample prediction and explanation commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from tulip.cli._context import _console, _emit_report, _tulip_errors, app
from tulip.core.types import Prediction, TaskType


def _load_and_predict(
    model_path: Path, text: str | None, audio: Path | None
) -> tuple[Any, Any, Prediction]:
    """Validate the single input, load the model, and classify; shared by predict and explain."""
    from tulip.core.exceptions import ConfigurationError
    from tulip.pipeline import DialectClassifier

    if (text is None) == (audio is None):
        raise ConfigurationError("provide exactly one input: a TEXT argument or --audio PATH")
    raw: Any = text if text is not None else audio
    classifier = DialectClassifier.load(model_path)
    _require_input_matches_task(classifier, text=text, audio=audio)
    prediction = classifier.predict(raw)
    return classifier, raw, prediction


@app.command()
@_tulip_errors
def predict(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    text: str | None = typer.Argument(None, help="Text to classify."),
    audio: Path | None = typer.Option(None, "--audio", help="Audio file to classify."),
    top_k: int = typer.Option(3, "--top-k", min=1, help="How many classes to display."),
    json_output: bool = typer.Option(False, "--json", help="Emit the prediction as JSON."),
    map_out: Path | None = typer.Option(
        None, "--map", help="Write an interactive prediction map to this HTML file."
    ),
    explain: str | None = typer.Option(
        None, "--explain", help="Explainer name (e.g. top_tfidf, lime, nearest_examples)."
    ),
    uncertainty: bool = typer.Option(
        False, "--uncertainty", help="Show the aleatoric/epistemic split (ensemble models only)."
    ),
) -> None:
    """Classify one text (argument) or one audio file (--audio)."""
    classifier, raw, prediction = _load_and_predict(model_path, text, audio)

    if json_output:
        _console.print_json(prediction.model_dump_json())
    else:
        _print_prediction(prediction, top_k)

    if uncertainty:
        _print_uncertainty(classifier, raw)
    if map_out is not None:
        _export_map(prediction, map_out)
    if explain is not None:
        _print_explanation(classifier, raw, explain)


@app.command()
@_tulip_errors
def explain(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    text: str | None = typer.Argument(None, help="Text to explain."),
    audio: Path | None = typer.Option(None, "--audio", help="Audio file to explain."),
    method: str = typer.Option(
        "top_tfidf",
        "--method",
        "-m",
        help="Explainer name (top_tfidf, lime, shap, nearest_examples).",
    ),
) -> None:
    """Explain one prediction: which features drove it, or its nearest neighbours.

    The architecture contract lists ``explain`` as its own command group; it is
    also reachable as ``predict --explain <method>``. This standalone form
    classifies the input and renders only the explanation.
    """
    classifier, raw, prediction = _load_and_predict(model_path, text, audio)
    _print_prediction(prediction, top_k=3)
    _print_explanation(classifier, raw, method)


@app.command("explain-global")
@_tulip_errors
def explain_global(
    data: Path = typer.Argument(..., help="A labelled corpus to summarise (jsonl/csv/...)."),
    level: str = typer.Option(
        "dialect", "--level", help="Gold label level for the lift axis: dialect | family."
    ),
    top_k: int = typer.Option(20, "--top-k", min=1, help="Phenomena to display."),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    out: Path | None = typer.Option(None, "--out", help="Also write the report JSON here."),
) -> None:
    """Summarise the dialectal evidence across a whole labelled corpus.

    Where ``explain`` justifies one prediction, this rolls the marker and
    isogloss evidence up over the corpus: which phenomena occur, which gold
    dialect their carriers belong to, and how concentrated that link is by
    class-conditional lift. A high-lift isogloss is one that genuinely separates
    a dialect. The evidence is resource-defined, so no model is needed and the
    report is the same regardless of which classifier you trained.
    """
    from tulip.core.exceptions import ConfigurationError
    from tulip.data import read_samples
    from tulip.explain import dataset_evidence
    from tulip.labels.taxonomy import LabelLevel

    try:
        label_level = LabelLevel(level)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in LabelLevel)
        raise ConfigurationError(f"unknown level {level!r}; use one of: {allowed}") from exc

    report = dataset_evidence(read_samples(data), level=label_level, name=str(data))
    _emit_report(
        report,
        json_output=json_output,
        out=out,
        saved_label="evidence report",
        markdown=report.to_markdown(top_k=top_k),
    )


@app.command()
@_tulip_errors
def contrast(
    data: Path = typer.Argument(..., help="A labelled corpus to analyse (jsonl/csv/...)."),
    dialect_a: str = typer.Argument(
        ..., help="First dialect label (positive log-odds favours it)."
    ),
    dialect_b: str = typer.Argument(..., help="Second dialect label."),
    level: str = typer.Option(
        "dialect", "--level", help="Gold label level the two labels are read at: dialect | family."
    ),
    top_k: int = typer.Option(
        10, "--top-k", min=1, help="Features shown per family and direction."
    ),
    min_support: int = typer.Option(
        5,
        "--min-support",
        min=1,
        help="Minimum documents a feature must occur in to be contrasted.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    out: Path | None = typer.Option(None, "--out", help="Also write the report JSON here."),
) -> None:
    """Rank the linguistic features that most distinguish two dialects.

    Answers the dialectology question a leaderboard cannot: which lexical markers,
    phonological isoglosses, and morphological endings separate one dialect from
    another, in which direction, and by how much. The analysis is model-free (it
    reads the gold-labelled text through the marker lexicon and isogloss rules) and
    reports each feature's smoothed log-odds effect size with a Holm-corrected
    two-proportion test. Use `--level family` to contrast, for example, silesian
    against standard.
    """
    from tulip.core.exceptions import ConfigurationError
    from tulip.data import read_samples
    from tulip.explain.contrast import contrast_dialects
    from tulip.labels.taxonomy import LabelLevel

    try:
        label_level = LabelLevel(level)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in LabelLevel)
        raise ConfigurationError(f"unknown level {level!r}; use one of: {allowed}") from exc

    report = contrast_dialects(
        list(read_samples(data)),
        dialect_a,
        dialect_b,
        level=label_level,
        min_support=min_support,
    )
    _emit_report(
        report,
        json_output=json_output,
        out=out,
        saved_label="contrast report",
        markdown=report.to_markdown(top_k=top_k),
    )


def _require_input_matches_task(classifier: Any, *, text: str | None, audio: Path | None) -> None:
    """Reject a text/audio input that mismatches the model's modality.

    Feeding an audio path to a text model (or vice versa) otherwise dies deep in
    the feature stack with an opaque ``ValueError``/``TypeError`` that escapes the
    ``TulipError`` boundary as a raw traceback. This turns it into one clean line.
    """
    from tulip.core.exceptions import ConfigurationError

    if text is not None and classifier.task is not TaskType.TEXT:
        raise ConfigurationError(
            f"this model classifies {classifier.task.value}, not text; pass --audio PATH"
        )
    if audio is not None and classifier.task is not TaskType.AUDIO:
        raise ConfigurationError(
            f"this model classifies {classifier.task.value}, not audio; pass a text argument"
        )


def _print_prediction(prediction: Prediction, top_k: int) -> None:
    """Render a prediction as a probability table."""
    if prediction.abstained:
        _console.print(
            f"[yellow]abstained[/yellow] (top confidence {prediction.confidence:.1%} "
            "below the model's threshold)"
        )
    else:
        _console.print(f"prediction: [bold]{prediction.label}[/bold] ({prediction.confidence:.1%})")
    table = Table(show_header=True)
    table.add_column(prediction.level.value)
    table.add_column("probability", justify="right")
    for entry in prediction.top_k(top_k):
        table.add_row(entry.label, f"{entry.probability:.1%}")
    _console.print(table)


def _print_uncertainty(classifier: Any, raw: Any) -> None:
    """Show the aleatoric/epistemic split for one input (ensemble models only)."""
    from tulip.evaluation import decompose_uncertainty, member_probabilities

    members = member_probabilities(classifier, [raw])
    total, aleatoric, epistemic = decompose_uncertainty(members)
    _console.print(
        f"uncertainty (nats over {members.shape[0]} members): "
        f"total [bold]{float(total[0]):.3f}[/bold], "
        f"aleatoric {float(aleatoric[0]):.3f}, epistemic {float(epistemic[0]):.3f}"
    )


def _export_map(prediction: Prediction, destination: Path) -> None:
    """Write the interactive prediction map (needs the viz extra)."""
    from tulip.viz.map import prediction_map, save_map

    save_map(prediction_map(prediction), destination)
    _console.print(f"[green]map written to {destination}[/green]")


def _print_explanation(classifier: Any, raw: Any, method: str) -> None:
    """Render an explanation's attributions and/or neighbours."""
    explanation = classifier.explain(raw, method=method)
    if explanation.attributions:
        table = Table(title=f"evidence ({explanation.method})")
        table.add_column("token")
        table.add_column("weight", justify="right")
        for attribution in explanation.top_attributions(10):
            colour = "green" if attribution.weight > 0 else "red"
            table.add_row(attribution.token, f"[{colour}]{attribution.weight:+.4f}[/{colour}]")
        _console.print(table)
    if explanation.neighbors:
        table = Table(title="most similar training examples")
        table.add_column("label")
        table.add_column("similarity", justify="right")
        table.add_column("text", overflow="fold")
        for neighbor in explanation.neighbors:
            table.add_row(neighbor.label or "?", f"{neighbor.similarity:.2f}", neighbor.text or "")
        _console.print(table)
