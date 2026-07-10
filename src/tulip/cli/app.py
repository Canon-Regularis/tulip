"""The ``tulip`` command-line interface.

One operator surface over the whole toolkit: dataset inspection and
preparation, training, benchmarking, evaluation, single-sample prediction
(with optional map export and explanations), and the HTTP service.

Every command is wrapped by :func:`_tulip_errors`, which catches library
errors (:class:`~tulip.core.exceptions.TulipError`) at the command boundary
and renders them as one clean red line; full tracebacks appear only under
``--verbose``. Heavy imports stay inside command bodies so ``tulip --help``
never pays for scikit-learn.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from tulip import __version__
from tulip.core.exceptions import TulipError
from tulip.core.types import Prediction
from tulip.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="tulip",
    help="Polish dialect detection from text, transcribed speech, and audio.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
datasets_app = typer.Typer(help="Inspect and prepare the source corpora.", no_args_is_help=True)
app.add_typer(datasets_app, name="data")
cards_app = typer.Typer(help="Render dataset and model cards.", no_args_is_help=True)
app.add_typer(cards_app, name="card")

_console = Console()
_errors = Console(stderr=True, style="bold red")
_logger = get_logger(__name__)

_state = {"verbose": False}

_CommandT = TypeVar("_CommandT", bound=Callable[..., None])


def _tulip_errors(command: _CommandT) -> _CommandT:
    """Decorate a command with the uniform TulipError boundary."""

    @functools.wraps(command)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            command(*args, **kwargs)
        except TulipError as exc:
            if _state["verbose"]:
                raise
            _errors.print(f"error: {exc}")
            raise typer.Exit(code=1) from exc

    return wrapper  # type: ignore[return-value]  # functools.wraps preserves the signature


def _print_version(value: bool) -> None:
    if value:
        _console.print(f"tulip {__version__}")
        raise typer.Exit()


@app.callback()
def _global_options(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging and tracebacks."),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version,
        is_eager=True,  # handled during parsing, before "missing command" validation
        help="Print the tulip version and exit.",
    ),
) -> None:
    _state["verbose"] = verbose
    configure_logging(logging.DEBUG if verbose else logging.WARNING)


# ------------------------------------------------------------------ datasets


@datasets_app.command("list")
@_tulip_errors
def datasets_list(
    root: Path = typer.Option(Path("data/raw"), help="Local corpora root directory."),
) -> None:
    """List the catalogued corpora, their tiers, and local availability."""
    from tulip.data import DATASETS, catalog

    table = Table(title="tulip dataset catalog", show_lines=False)
    table.add_column("name", style="bold")
    table.add_column("tier", justify="center")
    table.add_column("tasks")
    table.add_column("local", justify="center")
    table.add_column("url", overflow="fold")
    for info in catalog():
        loader = DATASETS.create(info.name)
        available = loader.is_available(root / info.name)
        table.add_row(
            info.name,
            str(info.tier),
            ", ".join(info.tasks),
            "[green]yes[/green]" if available else "[dim]no[/dim]",
            info.url,
        )
    _console.print(table)
    _console.print(f"[dim]local root: {root} -- acquisition notes: docs/datasets.md[/dim]")


@datasets_app.command("download")
@_tulip_errors
def data_download(
    names: list[str] | None = typer.Argument(
        None, help="Corpus names (see `tulip data list`); omit with --all for everything."
    ),
    all_datasets: bool = typer.Option(False, "--all", help="Acquire every catalogued corpus."),
    root: Path = typer.Option(Path("data/raw"), help="Local corpora root directory."),
    force: bool = typer.Option(False, "--force", help="Re-download corpora already present."),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Sample cap forwarded to downloaders that support it."
    ),
) -> None:
    """Download every corpus that has an automatic source; print exact steps for the rest.

    Most dialect corpora have no licence-clean bulk download, so full
    automation is impossible — this command does everything that can be done
    and tells you precisely what remains.
    """
    from tulip.core.exceptions import ConfigurationError
    from tulip.data import DownloadStatus, download_datasets

    if not names and not all_datasets:
        raise ConfigurationError("name at least one corpus, or pass --all")
    reports = download_datasets(
        names if names else None,
        root,
        force=force,
        options={"limit": limit} if limit is not None else None,
    )

    table = Table(title="corpus acquisition")
    table.add_column("corpus", style="bold")
    table.add_column("status", justify="center")
    table.add_column("where / what next", overflow="fold")
    status_styles = {
        DownloadStatus.DOWNLOADED: "[green]downloaded[/green]",
        DownloadStatus.ALREADY_PRESENT: "[green]present[/green]",
        DownloadStatus.MANUAL: "[yellow]manual[/yellow]",
        DownloadStatus.FAILED: "[red]failed[/red]",
    }
    for report in reports:
        table.add_row(report.name, status_styles[report.status], report.detail)
    _console.print(table)

    manual = [report for report in reports if report.status is DownloadStatus.MANUAL]
    if manual:
        _console.print(
            f"[yellow]{len(manual)} corpus(es) need manual steps above[/yellow] — "
            "full instructions: docs/datasets.md"
        )
    failed = [report for report in reports if report.status is DownloadStatus.FAILED]
    if failed:
        _console.print(
            f"[red]{len(failed)} download(s) failed[/red] — remediation steps are "
            "in the table above"
        )
        raise typer.Exit(code=1)


@datasets_app.command("prepare")
@_tulip_errors
def data_prepare(
    config_path: Path = typer.Argument(..., help="Experiment config YAML."),
    output: Path | None = typer.Option(
        None, help="Split output directory (default: <output_dir>/<name>/splits)."
    ),
) -> None:
    """Build speaker-disjoint splits (load, clean, dedup, split, persist)."""
    from tulip.config import load_experiment_config
    from tulip.data import DatasetBuilder

    config = load_experiment_config(config_path)
    destination = output or config.output_dir / config.name / "splits"
    splits = DatasetBuilder(config.data).build(
        config.split, target=config.target, output_dir=destination
    )
    for split_name, size in splits.sizes().items():
        _console.print(f"{split_name}: {size} samples")
    _console.print(f"[green]splits written to {destination}[/green]")


def _report_written_corpus(path: Path, *, title: str, unit: str) -> None:
    """Read a freshly written corpus back and print its per-class distribution.

    Both synthesize commands report from the artifact (not the in-memory spec),
    which doubles as proof that the written manifest is loadable.
    """
    from tulip.data.reading import read_samples

    counts: dict[str, int] = {}
    for sample in read_samples(path):
        key = sample.labels.dialect or sample.labels.family or "__unlabelled__"
        counts[key] = counts.get(key, 0) + 1

    table = Table(title=title)
    table.add_column("class", style="bold")
    table.add_column(unit, justify="right")
    for label in sorted(counts):
        table.add_row(label, str(counts[label]))
    _console.print(table)
    _console.print(f"[green]{sum(counts.values())} {unit} written to {path}[/green]")


@datasets_app.command("synthesize")
@_tulip_errors
def data_synthesize(
    out: Path = typer.Option(Path("data/raw/synthetic"), "--out", help="Destination directory."),
    seed: int = typer.Option(7, help="Generator seed; fixes the corpus byte for byte."),
    speakers: int = typer.Option(8, "--speakers", min=2, help="Speakers per class."),
    per_speaker: int = typer.Option(12, "--per-speaker", min=1, help="Samples per speaker."),
    noise: float = typer.Option(0.10, "--noise", min=0.0, max=1.0, help="Foreign-marker rate."),
    marker_dropout: float = typer.Option(
        0.20, "--marker-dropout", min=0.0, max=1.0, help="Share of samples with no lexical marker."
    ),
    standard: bool = typer.Option(True, "--standard/--no-standard", help="Emit a standard class."),
) -> None:
    """Write a linguistically-grounded synthetic corpus (no data acquisition needed).

    The corpus is generated in-process, so a fresh checkout can run `tulip
    train configs/synthetic_text.yaml` immediately. Raising --marker-dropout
    makes the task harder; at 0.0 every linear model saturates at 1.000.
    """
    from tulip.data.synthetic import SyntheticSpec, write_synthetic_manifest

    spec = SyntheticSpec(
        n_speakers_per_dialect=speakers,
        samples_per_speaker=per_speaker,
        include_standard=standard,
        noise_level=noise,
        marker_dropout=marker_dropout,
        seed=seed,
    )
    path = write_synthetic_manifest(spec, out)
    _report_written_corpus(path, title=f"synthetic corpus (seed={seed})", unit="samples")


@datasets_app.command("synthesize-audio")
@_tulip_errors
def data_synthesize_audio(
    out: Path = typer.Option(
        Path("data/raw/synthetic_audio"), "--out", help="Destination directory."
    ),
    seed: int = typer.Option(7, help="Generator seed; fixes the clips byte for byte."),
    speakers: int = typer.Option(8, "--speakers", min=2, help="Speakers per class."),
    per_speaker: int = typer.Option(6, "--per-speaker", min=1, help="Clips per speaker."),
    duration: float = typer.Option(0.8, "--duration", min=0.1, help="Clip length in seconds."),
) -> None:
    """Write a synthetic audio corpus (16 kHz mono WAV clips) with zero acquisition.

    Each class has a distinct acoustic fingerprint (pitch register, vowel-space
    formants, spectral tilt), so classical audio features (mfcc/pitch/formants)
    separate them. It is a benchmark fixture, not real speech. Run
    `tulip train configs/synthetic_audio.yaml` afterwards.
    """
    from tulip.data.synthetic_audio import AudioSyntheticSpec, write_synthetic_audio_manifest

    spec = AudioSyntheticSpec(
        n_speakers_per_dialect=speakers,
        samples_per_speaker=per_speaker,
        duration_s=duration,
        seed=seed,
    )
    path = write_synthetic_audio_manifest(spec, out)
    _report_written_corpus(path, title=f"synthetic audio corpus (seed={seed})", unit="clips")


@datasets_app.command("validate")
@_tulip_errors
def data_validate(
    manifest: Path = typer.Argument(..., help="Manifest file (CSV/TSV/JSONL) to validate."),
    audio_root: Path | None = typer.Option(
        None, "--audio-root", help="Base directory for relative audio paths."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Check a manifest's integrity; exit 1 on errors so CI can gate on it.

    Out-of-taxonomy labels are warnings, not errors: corpus-specific label
    strings are explicitly allowed to flow through the pipeline.
    """
    from tulip.data.validation import validate_manifest

    report = validate_manifest(manifest, audio_root=audio_root)
    if json_output:
        _console.print_json(report.model_dump_json())
    else:
        _console.print(report.to_markdown())
    if not report.ok:
        raise typer.Exit(code=1)


# --------------------------------------------------------------- train/eval


@app.command()
@_tulip_errors
def train(config_path: Path = typer.Argument(..., help="Experiment config YAML.")) -> None:
    """Run one experiment end to end (build data, train, evaluate, persist)."""
    from tulip.config import load_experiment_config
    from tulip.pipeline import run_experiment

    result = run_experiment(load_experiment_config(config_path))
    _console.print(result.summary())
    _console.print(f"[green]model saved to {result.model_path}[/green]")


@app.command()
@_tulip_errors
def benchmark(
    config_path: Path = typer.Argument(..., help="Base experiment config YAML."),
    model: list[str] = typer.Option(
        [], "--model", "-m", help="Model registry name (repeatable); default: config's model."
    ),
    split: str = typer.Option("test", help="Split shown in the comparison table."),
) -> None:
    """Compare several models over one identical frozen split."""
    from tulip.config import load_experiment_config
    from tulip.evaluation.benchmark import comparison_table
    from tulip.pipeline import run_benchmark

    config = load_experiment_config(config_path)
    results = run_benchmark(config, models=model or None)
    frame = comparison_table(results, split=split)
    table = Table(title=f"benchmark {config.name!r} ({split} split)")
    for column in frame.columns:
        table.add_column(str(column))
    for _, row in frame.iterrows():
        table.add_row(*(_format_cell(value) for value in row))
    _console.print(table)
    _console.print(f"[green]benchmark artifacts under {config.output_dir / config.name}[/green]")


@app.command()
@_tulip_errors
def leaderboard(
    suite_path: Path = typer.Argument(..., help="Leaderboard suite YAML (see benchmarks/)."),
    out: Path = typer.Option(Path("benchmarks/results"), "--out", help="Artifact root."),
    split: str = typer.Option("test", help="Split shown in the printed table."),
) -> None:
    """Regenerate the reproducible leaderboard for a whole suite of configs.

    ``leaderboard.md`` and ``provenance.json`` are deterministic: the same seeds
    reproduce them byte for byte, which is what makes the committed artifact an
    auditable benchmark rather than a snapshot.
    """
    from tulip.evaluation.benchmark import comparison_table
    from tulip.evaluation.leaderboard import load_suite, run_leaderboard, write_leaderboard

    suite = load_suite(suite_path)
    results = run_leaderboard(suite)
    destination = out / suite.name
    write_leaderboard(results, destination, suite=suite)

    frame = comparison_table(results, split=split, sort_by="f1_macro")
    table = Table(title=f"leaderboard {suite.name!r} ({split} split)")
    for column in frame.columns:
        table.add_column(str(column))
    for _, row in frame.iterrows():
        table.add_row(*(_format_cell(value) for value in row))
    _console.print(table)
    _console.print(f"[green]leaderboard written to {destination}[/green]")


@app.command()
@_tulip_errors
def selftrain(
    labeled: Path = typer.Argument(..., help="Labelled samples (split .jsonl or manifest)."),
    unlabeled: Path = typer.Argument(..., help="Unlabelled samples to pseudo-label."),
    model: str = typer.Option("logistic_regression", "--model", "-m", help="Model registry name."),
    feature: list[str] = typer.Option(
        [], "--feature", "-f", help="Feature registry name (repeatable)."
    ),
    raw: bool = typer.Option(
        False, "--raw", help="The model consumes raw text/audio itself (neural); pass no features."
    ),
    threshold: float = typer.Option(
        0.90, "--threshold", min=0.0, max=1.0, help="Minimum confidence to trust a pseudo-label."
    ),
    iters: int = typer.Option(3, "--iters", min=1, help="Maximum self-training rounds."),
    out: Path | None = typer.Option(None, "--out", help="Save the improved model here."),
) -> None:
    """Grow a classifier from a labelled seed set using confident pseudo-labels.

    This is what makes label-less corpora (e.g. `bigos`, which carries no
    dialect labels) contribute to training rather than sitting unused.
    """
    from tulip.core.exceptions import ConfigurationError
    from tulip.data import read_samples
    from tulip.pipeline.selftrain import SelfTrainConfig, self_train

    # A classical model handed raw strings dies deep inside sklearn with an
    # unreadable ValueError. The registry carries no "raw input" capability flag
    # to infer this from, so make the caller say which shape they meant.
    if raw and feature:
        raise ConfigurationError("--raw takes no --feature; drop one of them")
    if not raw and not feature:
        raise ConfigurationError(
            "no --feature given: classical models need at least one feature extractor "
            "(e.g. -f char_tfidf). Raw-input models (herbert, fasttext, wav2vec2, ...) "
            "take none -- pass --raw to say so explicitly."
        )

    result = self_train(
        labeled=list(read_samples(labeled)),
        unlabeled=list(read_samples(unlabeled)),
        model=model,
        features=list(feature),
        config=SelfTrainConfig(confidence_threshold=threshold, max_iterations=iters),
    )

    table = Table(title="self-training")
    table.add_column("round", justify="right")
    table.add_column("pseudo-labels added", justify="right")
    for index, added in enumerate(result.n_pseudo_per_iteration, start=1):
        table.add_row(str(index), str(added))
    _console.print(table)
    _console.print(
        f"converged after {result.iterations} round(s); "
        f"{len(result.pseudo_samples)} pseudo-label(s) total"
    )
    if out is not None:
        _console.print(f"[green]model saved to {result.classifier.save(out)}[/green]")


@app.command()
@_tulip_errors
def evaluate(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    data: Path = typer.Argument(
        ..., help="Labelled samples: split .jsonl, manifest file, or manifest directory."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Evaluate a saved model on labelled samples."""
    from tulip.data import read_samples
    from tulip.pipeline import DialectClassifier, evaluate_samples

    classifier = DialectClassifier.load(model_path)
    report = evaluate_samples(classifier, list(read_samples(data)), name=str(data))
    if json_output:
        _console.print_json(report.model_dump_json())
    else:
        _console.print(report.to_markdown())


# ------------------------------------------------------------------ predict


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
) -> None:
    """Classify one text (argument) or one audio file (--audio)."""
    from tulip.core.exceptions import ConfigurationError
    from tulip.pipeline import DialectClassifier

    if (text is None) == (audio is None):
        raise ConfigurationError("provide exactly one input: a TEXT argument or --audio PATH")
    raw: Any = text if text is not None else audio
    classifier = DialectClassifier.load(model_path)
    _require_input_matches_task(classifier, text=text, audio=audio)
    prediction = classifier.predict(raw)

    if json_output:
        _console.print_json(prediction.model_dump_json())
    else:
        _print_prediction(prediction, top_k)

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
    from tulip.core.exceptions import ConfigurationError
    from tulip.pipeline import DialectClassifier

    if (text is None) == (audio is None):
        raise ConfigurationError("provide exactly one input: a TEXT argument or --audio PATH")
    raw: Any = text if text is not None else audio
    classifier = DialectClassifier.load(model_path)
    _require_input_matches_task(classifier, text=text, audio=audio)
    prediction = classifier.predict(raw)
    _print_prediction(prediction, top_k=3)
    _print_explanation(classifier, raw, method)


def _require_input_matches_task(classifier: Any, *, text: str | None, audio: Path | None) -> None:
    """Reject a text/audio input that mismatches the model's modality.

    Feeding an audio path to a text model (or vice versa) otherwise dies deep in
    the feature stack with an opaque ``ValueError``/``TypeError`` that escapes the
    ``TulipError`` boundary as a raw traceback. This turns it into one clean line.
    """
    from tulip.core.exceptions import ConfigurationError
    from tulip.core.types import TaskType

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


# -------------------------------------------------------------------- cards


def _emit(markdown: str, destination: Path | None) -> None:
    """Print a rendered card, or write it to ``destination``."""
    if destination is None:
        _console.print(markdown)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown + "\n", encoding="utf-8", newline="\n")
    _console.print(f"[green]card written to {destination}[/green]")


def _read_json_mapping(path: Path, *, what: str) -> dict[str, Any]:
    """Read a JSON object, converting IO/parse failures into a clean ``DataError``.

    ``tulip.utils.io.read_json`` raises the underlying ``FileNotFoundError`` /
    ``JSONDecodeError``, and those escape the ``_tulip_errors`` boundary as a
    traceback that leaks absolute paths. Every other file-reading command in this
    CLI already fails with one clean ``error:`` line; this makes the card
    commands behave the same.
    """
    from tulip.core.exceptions import DataError
    from tulip.utils.io import read_json

    if not path.is_file():
        raise DataError(f"{what} not found: {path}")
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:  # JSONDecodeError subclasses ValueError
        raise DataError(f"{what} at {path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataError(f"{what} at {path} must be a JSON object, got {type(payload).__name__}")
    return payload


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
        # Not every dataset is catalogued -- the generic `manifest` loader
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


# -------------------------------------------------------------------- serve


@app.command()
@_tulip_errors
def serve(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Bind port."),
) -> None:
    """Serve the model over HTTP (text + audio upload; needs the serve extra)."""
    from tulip.serve.app import create_app
    from tulip.utils.optional import optional_import

    uvicorn = optional_import("uvicorn", extra="serve", purpose="the HTTP service")
    uvicorn.run(create_app(model_path), host=host, port=port)


# ------------------------------------------------------------------ helpers


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    """Console entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
