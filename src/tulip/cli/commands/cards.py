"""Dataset and model card commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from tulip.cli._context import _console, _read_json_mapping, _tulip_errors, cards_app

if TYPE_CHECKING:
    from tulip.core.types import DatasetInfo


def _emit(markdown: str, destination: Path | None) -> None:
    """Print a rendered card, or write it to ``destination``."""
    if destination is None:
        _console.print(markdown)
        return
    from tulip._serialize import write_markdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(destination, markdown)
    _console.print(f"[green]card written to {destination}[/green]")


def _resolve_dataset_info(name: str | None, sources: list[str], *, hint: str = "") -> DatasetInfo:
    """Resolve corpus metadata, inferring the name from a single source when unset.

    When ``name`` is not given the corpus must be unambiguous: exactly one source,
    otherwise a clean error asks for ``--dataset``. Not every corpus is catalogued
    (the generic manifest loader is not), so an uncatalogued name falls back to the
    loader's own metadata rather than a fabricated tier and licence; an unknown name
    still raises.
    """
    from tulip.core.exceptions import ConfigurationError, DataError
    from tulip.data import DATASETS, get_dataset_info

    if name is None:
        if len(sources) != 1:
            raise ConfigurationError(
                f"cannot infer the corpus from {len(sources)} source(s) {sources}; "
                f"pass --dataset NAME{hint}"
            )
        name = sources[0]
    try:
        return get_dataset_info(name)
    except DataError:
        return DATASETS.create(name).info


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
    from tulip.evaluation.cards import dataset_card

    manifest = _read_json_mapping(build_manifest, what="build manifest")
    sources = list(manifest.get("sources") or {})
    info = _resolve_dataset_info(dataset, sources)
    _emit(dataset_card(info, manifest), out)


@cards_app.command("datasheet")
@_tulip_errors
def card_datasheet(
    build_dir: Path = typer.Argument(
        ..., help="A build directory (train/validation/test.jsonl) from `data prepare`."
    ),
    spec: Path = typer.Option(
        ..., "--spec", help="Datasheet prose spec YAML (see benchmarks/datasheets/)."
    ),
    dataset: str | None = typer.Option(
        None, "--dataset", help="Catalogued corpus name (default: inferred from the samples)."
    ),
    manifest: Path | None = typer.Option(
        None, "--manifest", help="Source manifest to embed a `data validate` conformance section."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Write the datasheet here instead of stdout."
    ),
) -> None:
    """Render a Gebru-style datasheet from a built dataset and a prose spec.

    Composes the corpus's catalog metadata, its split/speaker/class distributions
    at every taxonomy level, its geographic and demographic composition, and the
    prose fields in ``--spec`` into one byte-stable document. For a benchmark that
    merges corpora, render one datasheet per source corpus.
    """
    from tulip.data.splitting import load_splits
    from tulip.evaluation.datasheet import datasheet, load_datasheet_spec

    splits = load_splits(build_dir)
    spec_model = load_datasheet_spec(spec)
    sources = sorted({s.source for group in splits.as_dict().values() for s in group})
    info = _resolve_dataset_info(dataset, sources, hint=" (render one datasheet per source corpus)")

    conformance = None
    if manifest is not None:
        from tulip.data.validation import validate_manifest

        conformance = validate_manifest(manifest).to_markdown()
    _emit(datasheet(info, splits, spec_model, conformance=conformance), out)


@cards_app.command("benchmark")
@_tulip_errors
def card_benchmark(
    board_dir: Path = typer.Argument(
        ..., help="A leaderboard output directory (leaderboard.md, significance-*.md)."
    ),
    datasheet: Path | None = typer.Option(
        None, "--datasheet", help="A rendered datasheet markdown to embed as the Dataset section."
    ),
    bias: Path | None = typer.Option(
        None,
        "--bias",
        help="A fairness/bias analysis markdown to embed (from `analyze --fairness`).",
    ),
    title: str | None = typer.Option(None, "--title", help="Override the report title."),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Stamp a 'synthetic fixture, not real accuracy' caption."
    ),
    out: Path | None = typer.Option(None, "--out", help="Write the report here instead of stdout."),
) -> None:
    """Assemble a paper-style benchmark report from a board plus an optional datasheet.

    Composes the label hierarchy, the protocol, the committed ``leaderboard.md`` and
    ``significance-*.md`` from ``board_dir``, and any supplied datasheet/bias
    sections into one byte-stable document (the seed for ``docs/benchmark.md``).
    """
    from tulip.evaluation.benchmark_report import DEFAULT_TITLE, benchmark_report

    datasheet_md = datasheet.read_text(encoding="utf-8") if datasheet is not None else None
    bias_md = bias.read_text(encoding="utf-8") if bias is not None else None
    report = benchmark_report(
        board_dir,
        title=title or DEFAULT_TITLE,
        datasheet_md=datasheet_md,
        bias_md=bias_md,
        synthetic=synthetic,
    )
    _emit(report, out)


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
