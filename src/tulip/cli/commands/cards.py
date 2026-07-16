"""Dataset and model card commands."""

from __future__ import annotations

from pathlib import Path

import typer

from tulip.cli._context import _console, _read_json_mapping, _tulip_errors, cards_app


def _emit(markdown: str, destination: Path | None) -> None:
    """Print a rendered card, or write it to ``destination``."""
    if destination is None:
        _console.print(markdown)
        return
    from tulip._serialize import write_markdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(destination, markdown)
    _console.print(f"[green]card written to {destination}[/green]")


@cards_app.command("dataset")
@_tulip_errors
def card_dataset(
    build_manifest: Path = typer.Argument(..., help="A build_manifest.json from `data prepare`."),
    dataset: str | None = typer.Option(
        None, "--dataset", help="Catalogued corpus name (default: inferred from the manifest)."
    ),
    out: Path | None = typer.Option(None, "--out", help="Write the card here instead of stdout."),
) -> None:
    """Render a dataset card from a built dataset's audit manifest."""
    from tulip.core.exceptions import ConfigurationError, DataError
    from tulip.data import DATASETS, get_dataset_info
    from tulip.evaluation.cards import dataset_card

    manifest = _read_json_mapping(build_manifest, what="build manifest")
    name = dataset
    if name is None:
        sources = list(manifest.get("sources") or {})
        if len(sources) != 1:
            raise ConfigurationError(
                f"cannot infer the corpus from {len(sources)} source(s) {sources}; "
                "pass --dataset NAME"
            )
        name = sources[0]

    try:
        info = get_dataset_info(name)
    except DataError:
        # Not every dataset is catalogued: the generic `manifest` loader
        # deliberately is not. Ask the loader for its own metadata rather than
        # fabricating a tier and a licence. An unknown name still raises.
        info = DATASETS.create(name).info
    _emit(dataset_card(info, manifest), out)


@cards_app.command("model")
@_tulip_errors
def card_model(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    report: list[Path] = typer.Option(
        [], "--report", help="Evaluation report JSON (repeatable); default: auto-discovered."
    ),
    out: Path | None = typer.Option(None, "--out", help="Write the card here instead of stdout."),
) -> None:
    """Render a model card from a saved model's metadata and its evaluation reports.

    Reports are read from disk rather than from the model artifact: `save_model`
    deliberately does not embed metrics, so the card pairs `metadata.json` with
    the sibling `report_<split>.json` files written by `tulip train`.
    """
    from pydantic import ValidationError

    from tulip.core.exceptions import DataError
    from tulip.evaluation.cards import model_card
    from tulip.evaluation.report import EvaluationReport
    from tulip.models.persistence import METADATA_FILENAME

    sidecar = _read_json_mapping(model_path / METADATA_FILENAME, what="model metadata")
    paths = list(report) or sorted(model_path.parent.glob("report_*.json"))
    reports = {}
    for path in paths:
        _read_json_mapping(path, what="evaluation report")  # clean error on missing/corrupt JSON
        try:
            reports[path.stem.removeprefix("report_")] = EvaluationReport.load(path)
        except ValidationError as exc:
            raise DataError(
                f"{path} is not a valid evaluation report ({exc.error_count()} schema error(s)); "
                "expected a file written by `tulip train`"
            ) from exc
    _emit(model_card(sidecar, reports), out)
