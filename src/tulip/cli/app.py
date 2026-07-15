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
from rich.markup import escape
from rich.table import Table

from tulip import __version__
from tulip.core.exceptions import TulipError
from tulip.core.types import Prediction, TaskType
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
repro_app = typer.Typer(
    help="Reproducibility checks for the committed leaderboard.", no_args_is_help=True
)
app.add_typer(repro_app, name="repro")
registry_app = typer.Typer(
    help="Content-addressed model registry (versioning, promotion, rollback).",
    no_args_is_help=True,
)
app.add_typer(registry_app, name="registry")
models_app = typer.Typer(help="Inspect the model registry.", no_args_is_help=True)
app.add_typer(models_app, name="models")
features_app = typer.Typer(help="Inspect the feature registry.", no_args_is_help=True)
app.add_typer(features_app, name="features")
explainers_app = typer.Typer(help="Inspect the explainer registry.", no_args_is_help=True)
app.add_typer(explainers_app, name="explainers")

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
    _console.print(f"[dim]local root: {root}; acquisition notes: docs/datasets.md[/dim]")


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
    automation is impossible. This command does everything that can be done
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
            f"[yellow]{len(manual)} corpus(es) need manual steps above[/yellow]; "
            "full instructions: docs/datasets.md"
        )
    failed = [report for report in reports if report.status is DownloadStatus.FAILED]
    if failed:
        _console.print(
            f"[red]{len(failed)} download(s) failed[/red]; remediation steps are in the table above"
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
    _print_frame(frame, f"benchmark {config.name!r} ({split} split)")
    _console.print(f"[green]benchmark artifacts under {config.output_dir / config.name}[/green]")


@app.command()
@_tulip_errors
def leaderboard(
    suite_path: Path = typer.Argument(..., help="Leaderboard suite YAML (see benchmarks/)."),
    out: Path = typer.Option(Path("benchmarks/results"), "--out", help="Artifact root."),
    split: str = typer.Option("test", help="Split shown in the printed table."),
    tracks: bool = typer.Option(
        False, "--tracks", help="Read a multi-track suite and rank each modality separately."
    ),
    significance: bool = typer.Option(
        True,
        "--significance/--no-significance",
        help="Also write per-experiment paired significance (CIs + McNemar).",
    ),
    sig_seed: int = typer.Option(0, "--sig-seed", help="Bootstrap seed for the significance CIs."),
) -> None:
    """Regenerate the reproducible leaderboard for a whole suite of configs.

    ``leaderboard.md`` and ``provenance.json`` are deterministic: the same seeds
    reproduce them byte for byte, which is what makes the committed artifact an
    auditable benchmark rather than a snapshot. ``--tracks`` reads a multi-track
    suite instead and writes one ranking per modality (text, audio, transcribed
    speech) to ``leaderboard-tracks.md``.
    """
    if tracks:
        _run_tracked_leaderboard(suite_path, out)
        return

    from tulip.evaluation.benchmark import comparison_table
    from tulip.evaluation.leaderboard import (
        load_suite,
        run_leaderboard,
        write_leaderboard,
        write_significance,
    )

    suite = load_suite(suite_path)
    results = run_leaderboard(suite)
    destination = out / suite.name
    write_leaderboard(results, destination, suite=suite)
    if significance:
        experiments = write_significance(results, destination, seed=sig_seed)
        if experiments:
            _console.print(
                f"[green]significance written for {len(experiments)} experiment(s)[/green]"
            )

    frame = comparison_table(results, split=split, sort_by="f1_macro")
    _print_frame(frame, f"leaderboard {suite.name!r} ({split} split)")
    _console.print(f"[green]leaderboard written to {destination}[/green]")


def _run_tracked_leaderboard(suite_path: Path, out: Path) -> None:
    """Run and write a multi-track leaderboard (``leaderboard --tracks``)."""
    from tulip.evaluation.tracks import (
        load_tracked_suite,
        run_tracked_leaderboard,
        write_tracked_leaderboard,
    )

    suite = load_tracked_suite(suite_path)
    track_results = run_tracked_leaderboard(suite)
    destination = out / suite.name
    write_tracked_leaderboard(track_results, destination, suite=suite)
    for track_result in track_results:
        _console.print(
            f"[dim]track {track_result.track.name}: {len(track_result.results)} result(s)[/dim]"
        )
    _console.print(f"[green]tracked leaderboard written to {destination}[/green]")


@app.command()
@_tulip_errors
def efficiency(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    data: Path = typer.Argument(..., help="Samples to time predictions over."),
    model: str = typer.Option("model", "--model", help="Model name recorded on the record."),
    repeats: int = typer.Option(3, "--repeats", min=1, help="Timed passes; the median is kept."),
    out: Path | None = typer.Option(None, "--out", help="Write the record JSON here (excluded)."),
    json_output: bool = typer.Option(False, "--json", help="Emit the record as JSON."),
) -> None:
    """Measure a saved model's inference latency, size, and parameter count.

    These numbers are machine dependent, so the record is an excluded artifact:
    it never feeds the byte-stable leaderboard or provenance.
    """
    from tulip.data import read_samples
    from tulip.evaluation.efficiency import measure_efficiency, write_efficiency
    from tulip.pipeline import DialectClassifier

    classifier = DialectClassifier.load(model_path)
    record = measure_efficiency(
        classifier,
        list(read_samples(data)),
        model=model,
        repeats=repeats,
        model_dir=model_path,
    )
    if out is not None:
        write_efficiency([record], out)
        _console.print(f"[green]efficiency written to {out}[/green]")
    if json_output:
        _console.print_json(record.model_dump_json())
    else:
        size = (
            "n/a"
            if record.model_size_bytes is None
            else f"{record.model_size_bytes / 1024:.1f} KiB"
        )
        params = "n/a" if record.n_params is None else str(record.n_params)
        _console.print(
            f"latency [bold]{record.latency_ms:.4f} ms[/bold]/sample over {record.n_samples} "
            f"sample(s); size {size}; params {params}"
        )


@repro_app.command("verify")
@_tulip_errors
def repro_verify(
    suite_path: Path = typer.Argument(..., help="Leaderboard suite YAML to regenerate."),
    against: Path = typer.Option(
        Path("benchmarks/results"), "--against", help="Committed artifact root to compare with."
    ),
) -> None:
    """Regenerate a suite and fail if it no longer matches the committed board.

    Byte-compares every committed artifact except ``leaderboard.json`` (which
    carries wall-clock timings). Exits non-zero on any drift, so CI catches a
    hidden nondeterminism, a moved generator default, an environment shift, or a
    silent metric regression instead of shipping it.
    """
    import tempfile

    from tulip.cli._repro import verify_reproduction

    with tempfile.TemporaryDirectory(prefix="tulip-repro-") as scratch:
        drift = verify_reproduction(suite_path, against, Path(scratch))
    if drift:
        _errors.print(f"reproduction FAILED: {len(drift)} artifact(s) drifted:")
        for line in drift:
            _errors.print(f"  - {line}")
        raise typer.Exit(code=1)
    _console.print("[green]reproduced the committed leaderboard byte-for-byte[/green]")


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


_DEFAULT_REGISTRY_ROOT = Path("artifacts/registry")


def _registry_option() -> Path:
    return typer.Option(_DEFAULT_REGISTRY_ROOT, "--registry", help="Registry root directory.")


@registry_app.command("add")
@_tulip_errors
def registry_add(
    model_dir: Path = typer.Argument(..., help="Saved model directory to register."),
    name: str = typer.Option(..., "--name", help="Model name in the registry."),
    version: str = typer.Option(..., "--version", help="Version label (e.g. 1, 2025-01-a)."),
    stage: str = typer.Option("staging", "--stage", help="staging | production | archived."),
    report: Path | None = typer.Option(
        None, "--report", help="Evaluation report JSON whose headline metrics to record."
    ),
    registry: Path = _registry_option(),
) -> None:
    """Register a saved model artifact under a name and version (content-addressed)."""
    from tulip.deploy import ModelRegistry, Stage

    metrics = _report_metrics(report) if report is not None else None
    entry = ModelRegistry(registry).add(
        model_dir, name=name, version=version, stage=Stage(stage), metrics=metrics
    )
    _console.print(
        f"[green]registered {entry.name}@{entry.version}[/green] "
        f"({entry.stage.value}) digest={entry.digest[:12]}"
    )


@registry_app.command("promote")
@_tulip_errors
def registry_promote(
    name: str = typer.Argument(..., help="Model name."),
    version: str = typer.Argument(..., help="Version to promote."),
    stage: str = typer.Option("production", "--stage", help="Target stage."),
    registry: Path = _registry_option(),
) -> None:
    """Promote a version to a stage (production archives the previous production)."""
    from tulip.deploy import ModelRegistry, Stage

    entry = ModelRegistry(registry).promote(name, version, stage=Stage(stage))
    _console.print(f"[green]{entry.name}@{entry.version} is now {entry.stage.value}[/green]")


@registry_app.command("rollback")
@_tulip_errors
def registry_rollback(
    name: str = typer.Argument(..., help="Model name to roll back."),
    registry: Path = _registry_option(),
) -> None:
    """Restore the previous production version of a model in one step."""
    from tulip.deploy import ModelRegistry

    entry = ModelRegistry(registry).rollback(name)
    _console.print(f"[green]rolled back: {entry.name}@{entry.version} is now production[/green]")


@registry_app.command("ls")
@_tulip_errors
def registry_ls(
    name: str | None = typer.Argument(None, help="Filter to one model name."),
    registry: Path = _registry_option(),
) -> None:
    """List registered model versions and their stages."""
    from tulip.deploy import ModelRegistry

    store = ModelRegistry(registry)
    entries = store.versions_of(name) if name else store.entries()
    table = Table(title=f"model registry ({registry})")
    for column in ("name", "version", "stage", "digest", "target", "model class"):
        table.add_column(column)
    for entry in entries:
        table.add_row(
            entry.name,
            entry.version,
            entry.stage.value,
            entry.digest[:12],
            entry.target or "-",
            entry.model_class.rsplit(".", 1)[-1],
        )
    _console.print(table)


@registry_app.command("resolve")
@_tulip_errors
def registry_resolve(
    reference: str = typer.Argument(..., help="name, name@production, or name@<version>."),
    registry: Path = _registry_option(),
) -> None:
    """Resolve a reference to its artifact path and digest."""
    from tulip.deploy import ModelRegistry

    store = ModelRegistry(registry)
    entry = store.resolve(reference)
    _console.print(f"{entry.name}@{entry.version} ({entry.stage.value})")
    _console.print(f"digest: {entry.digest}")
    _console.print(f"path:   {store.path_for(entry)}")


def _report_metrics(report_path: Path) -> dict[str, float]:
    """Read headline metrics from an evaluation report JSON for a registry entry."""
    payload = _read_json_mapping(report_path, what="evaluation report")
    keys = ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc_macro_ovr")
    return {key: float(payload[key]) for key in keys if isinstance(payload.get(key), (int, float))}


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
            "take none; pass --raw to say so explicitly."
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
def crossval(
    config_path: Path = typer.Argument(..., help="Experiment config YAML."),
    k: int = typer.Option(5, "--k", min=2, help="Number of folds."),
    seeds: str = typer.Option("0", "--seeds", help="Comma-separated fold seeds (e.g. 0,1,2)."),
) -> None:
    """Grouped, stratified K-fold cross-validation with multi-seed aggregation.

    Reports each metric's mean and 95% confidence interval across all folds, so a
    single lucky split cannot flatter the model. Folds are speaker-disjoint.
    """
    from tulip.config import load_experiment_config
    from tulip.pipeline import CVConfig, run_cross_validation

    config = load_experiment_config(config_path)
    seed_tuple = tuple(int(part) for part in seeds.split(",") if part.strip())
    report = run_cross_validation(config, CVConfig(k=k, seeds=seed_tuple))

    table = Table(title=f"cross-validation {config.model.name!r} ({report.target})")
    for column in ("metric", "mean", "std", "95% CI"):
        table.add_column(column)
    for metric in report.metrics:
        table.add_row(
            metric.metric,
            f"{metric.mean:.4f}",
            f"{metric.std:.4f}",
            f"[{metric.low:.4f}, {metric.high:.4f}]",
        )
    _console.print(table)
    _console.print(
        f"[dim]{len(report.folds)} fold runs ({k}-fold x {len(seed_tuple)} seed(s))[/dim]"
    )


@app.command()
@_tulip_errors
def transfer(
    config_path: Path = typer.Argument(..., help="Experiment config YAML (multi-corpus data)."),
    matrix: bool = typer.Option(
        False, "--matrix", help="Full train-by-test transfer matrix instead of leave-one-out."
    ),
) -> None:
    """Cross-corpus transfer: does the model learn dialect or corpus artifacts?

    Partitions the data by source corpus. By default runs leave-one-corpus-out
    (train on the rest, test on the held-out corpus). With ``--matrix`` fills the
    full train-corpus by test-corpus grid.
    """
    from tulip.config import load_experiment_config
    from tulip.evaluation import run_loco, transfer_matrix

    config = load_experiment_config(config_path)
    report = transfer_matrix(config) if matrix else run_loco(config)
    _console.print(report.to_markdown())


@app.command()
@_tulip_errors
def robustness(
    config_path: Path = typer.Argument(..., help="Experiment config YAML."),
    perturbation: list[str] | None = typer.Option(
        None,
        "--perturbation",
        "-p",
        help="Perturbation name, repeatable (default dialect_intensity_dial). "
        "Options: dialect_intensity_dial, standardize, asr_noise, typo_noise.",
    ),
    levels: str = typer.Option(
        "0,0.25,0.5,0.75,1.0", "--levels", help="Comma-separated intensity levels in [0, 1]."
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the perturbation draws."),
    out: Path | None = typer.Option(
        None, "--out", help="Directory to write robustness-<name>.md and .json."
    ),
) -> None:
    """Score a model as its inputs are perturbed along a linguistic intensity axis.

    Trains once on the clean split, then re-scores the test split perturbed at
    each level. The grounded perturbations (dialect_intensity_dial, standardize)
    move text along the standard-to-dialect axis; asr_noise and typo_noise stress
    the surface channel.
    """
    from tulip._serialize import write_markdown
    from tulip.config import load_experiment_config
    from tulip.core.exceptions import ConfigurationError
    from tulip.robustness import PerturbationConfig, run_robustness

    level_tuple = tuple(float(part) for part in levels.split(",") if part.strip())
    if not level_tuple or any(not 0.0 <= level <= 1.0 for level in level_tuple):
        raise ConfigurationError("--levels must be non-empty and within [0, 1]")
    names = perturbation or ["dialect_intensity_dial"]
    specs = [PerturbationConfig(name=name, levels=level_tuple, seed=seed) for name in names]

    config = load_experiment_config(config_path)
    report = run_robustness(config, perturbations=specs)
    _console.print(report.to_markdown())
    if out is not None:
        write_markdown(out / f"robustness-{config.name}.md", report.to_markdown())
        report.save(out / f"robustness-{config.name}.json")
        _console.print(f"[green]wrote robustness artifacts to {out}[/green]")


@app.command()
@_tulip_errors
def conformal(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    calibration: Path = typer.Argument(..., help="Held-out calibration samples."),
    test: Path = typer.Argument(..., help="Test samples to measure coverage on."),
    alpha: float = typer.Option(0.1, "--alpha", min=0.0, max=1.0, help="Miscoverage rate."),
    mondrian: bool = typer.Option(False, "--mondrian", help="Per-class (class-conditional) sets."),
) -> None:
    """Calibrate prediction sets and report their coverage.

    Fits split conformal on the calibration split, then measures empirical
    coverage and mean set size on the test split. Coverage should meet the
    ``1 - alpha`` target.
    """
    from tulip.data import read_samples
    from tulip.pipeline import ConformalClassifier, DialectClassifier

    classifier = DialectClassifier.load(model_path)
    conformal_classifier = ConformalClassifier(classifier, alpha=alpha, mondrian=mondrian)
    conformal_classifier.fit_conformal(list(read_samples(calibration)))
    report = conformal_classifier.evaluate_coverage(list(read_samples(test)))
    kind = "Mondrian" if mondrian else "marginal"
    _console.print(
        f"{kind} conformal (alpha={alpha}): coverage "
        f"[bold]{report.coverage:.3f}[/bold] (target {report.target_coverage:.2f}), "
        f"mean set size {report.mean_set_size:.2f} over {report.n_samples} samples"
    )


@app.command()
@_tulip_errors
def openset(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    calibration: Path = typer.Argument(..., help="Held-out calibration samples."),
    test: Path = typer.Argument(..., help="Test samples, possibly including unseen dialects."),
    alpha: float = typer.Option(0.1, "--alpha", min=0.0, max=1.0, help="Miscoverage rate."),
    mondrian: bool = typer.Option(False, "--mondrian", help="Per-class conformal thresholds."),
) -> None:
    """Flag inputs unlike any known dialect, and report open-set quality.

    Fits split conformal on the calibration split, then evaluates novelty
    detection on the test split. A test sample whose gold dialect was never
    trained on counts as truly novel, which is the deployment question of
    meeting a new region.
    """
    from tulip.data import read_samples
    from tulip.pipeline import ConformalClassifier, DialectClassifier, OpenSetClassifier

    classifier = DialectClassifier.load(model_path)
    conformal = ConformalClassifier(classifier, alpha=alpha, mondrian=mondrian)
    conformal.fit_conformal(list(read_samples(calibration)))
    report = OpenSetClassifier(conformal).evaluate(list(read_samples(test)))
    _console.print(report.to_markdown())


@app.command()
@_tulip_errors
def acquire(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    unlabeled: Path = typer.Argument(..., help="Unlabeled pool: split .jsonl or manifest."),
    strategy: str = typer.Option(
        "entropy",
        "--strategy",
        help="Acquisition strategy name; an unknown value lists the registered options.",
    ),
    budget: int | None = typer.Option(
        None, "--budget", min=1, help="Keep only the top-N candidates (default all)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the ranking as JSON."),
) -> None:
    """Rank an unlabeled pool by which samples to label first.

    A model trained on a labeled seed set scores each unlabeled sample by an
    acquisition strategy, so a fixed annotation budget buys the most signal. The
    dialect-aware ``intensity_gated`` strategy keeps budget off standard Polish
    the model merely happens to be unsure about. Ranking only; labeling is a
    human step.
    """
    from tulip.core.exceptions import ConfigurationError
    from tulip.core.registry import UnknownComponentError
    from tulip.data import read_samples
    from tulip.pipeline import STRATEGIES, DialectClassifier, rank_for_labeling

    classifier = DialectClassifier.load(model_path)
    try:
        candidates = rank_for_labeling(
            classifier, list(read_samples(unlabeled)), strategy=strategy, budget=budget
        )
    except UnknownComponentError as exc:
        # The valid set is derived from the registry, never a hardcoded list, so a
        # newly registered strategy is discoverable without editing the CLI.
        options = ", ".join(STRATEGIES.names())
        raise ConfigurationError(f"unknown strategy {strategy!r}; choose from: {options}") from exc
    if json_output:
        _console.print_json(data=[candidate.model_dump() for candidate in candidates])
        return
    table = Table(title=f"acquisition ranking ({strategy})")
    table.add_column("#", justify="right")
    table.add_column("sample")
    table.add_column("predicted")
    table.add_column("confidence", justify="right")
    table.add_column("score", justify="right")
    for rank, candidate in enumerate(candidates, start=1):
        table.add_row(
            str(rank),
            candidate.sample_id,
            candidate.predicted_label,
            f"{candidate.confidence:.1%}",
            f"{candidate.score:.4f}",
        )
    _console.print(table)


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
    if out is not None:
        report.save(out)
        _console.print(f"[green]evidence report written to {out}[/green]")
    if json_output:
        _console.print_json(report.model_dump_json())
    else:
        _console.print(report.to_markdown(top_k=top_k))


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


# -------------------------------------------------------------------- cards


def _emit(markdown: str, destination: Path | None) -> None:
    """Print a rendered card, or write it to ``destination``."""
    if destination is None:
        _console.print(markdown)
        return
    from tulip._serialize import write_markdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(destination, markdown)
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


# -------------------------------------------------------------------- serve


@app.command()
@_tulip_errors
def serve(
    model: Path = typer.Argument(
        ..., help="Saved model directory, or a registry reference (e.g. dialect@production)."
    ),
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Bind port."),
    registry: Path | None = typer.Option(
        None, "--registry", help="Registry root; then MODEL is a reference resolved from it."
    ),
) -> None:
    """Serve the model over HTTP (text + audio upload; needs the serve extra).

    Guards (auth, rate limit, concurrency, body-size cap, CORS, security headers)
    are read from ``TULIP_SERVE_*`` environment variables. With ``--registry`` the
    MODEL argument is a registry reference and the response carries
    ``X-Model-Version`` / ``X-Model-Digest``.
    """
    from tulip.deploy import ModelRegistry, artifact_digest
    from tulip.serve.app import create_app
    from tulip.utils.optional import optional_import

    if registry is not None:
        store = ModelRegistry(registry)
        entry = store.resolve(str(model))
        path, version, digest = store.path_for(entry), entry.version, entry.digest
    else:
        path, version, digest = model, None, artifact_digest(model)

    uvicorn = optional_import("uvicorn", extra="serve", purpose="the HTTP service")
    app_instance = create_app(path, model_version=version, model_digest=digest)
    uvicorn.run(app_instance, host=host, port=port)


# ------------------------------------------------------------- doctor & discovery


@app.command()
@_tulip_errors
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Report what runs on this install and what to pip install to unblock the rest."""
    from tulip.cli._doctor import run_doctor

    report = run_doctor()
    if json_output:
        _console.print_json(report.model_dump_json())
        return

    _console.print(
        f"[bold]tulip {report.tulip_version}[/bold] | "
        f"Python {report.python_version} | {report.platform}"
    )
    extras = Table(title="optional extras", show_lines=False)
    extras.add_column("extra", style="bold")
    extras.add_column("installed", justify="center")
    extras.add_column("unlocks")
    extras.add_column("install", overflow="fold")
    for extra in report.extras:
        extras.add_row(
            extra.name,
            "[green]yes[/green]" if extra.installed else "[dim]no[/dim]",
            extra.purpose,
            "" if extra.installed else escape(extra.install_hint),
        )
    _console.print(extras)

    runnable = report.runnable_count
    _console.print(
        f"[green]{runnable}[/green] of {len(report.components)} components runnable now."
    )
    if report.blocked_components:
        _console.print(
            f"[dim]{len(report.blocked_components)} blocked; install the extras above, or run "
            "`tulip models/features/explainers list` for the per-component breakdown.[/dim]"
        )


def _render_component_list(title: str, kinds: tuple[str, ...]) -> None:
    """Print an availability table for one or more registry kinds."""
    from tulip.cli._doctor import component_statuses

    rows = [status for status in component_statuses() if status.kind in kinds]
    show_kind = len({status.kind for status in rows}) > 1
    table = Table(title=title, show_lines=False)
    table.add_column("name", style="bold")
    if show_kind:
        table.add_column("kind")
    table.add_column("needs", justify="center")
    table.add_column("ready", justify="center")
    for status in rows:
        cells = [status.name]
        if show_kind:
            cells.append(status.kind)
        cells.append(status.extra or "core")
        cells.append("[green]yes[/green]" if status.available else "[dim]no[/dim]")
        table.add_row(*cells)
    _console.print(table)


@models_app.command("list")
@_tulip_errors
def models_list() -> None:
    """List the registered models and whether each runs on this install."""
    _render_component_list("models", ("model",))


@features_app.command("list")
@_tulip_errors
def features_list() -> None:
    """List the registered text and audio feature extractors."""
    _render_component_list("features", ("text feature", "audio feature"))


@explainers_app.command("list")
@_tulip_errors
def explainers_list() -> None:
    """List the registered explainers and whether each runs on this install."""
    _render_component_list("explainers", ("explainer",))


# ------------------------------------------------------------------ citation


@app.command()
@_tulip_errors
def cite(
    citation_format: str = typer.Option(
        "bibtex", "--format", "-f", help="Citation format: bibtex or apa."
    ),
    check: bool = typer.Option(
        False, "--check", help="Verify version parity across the metadata files; exit 1 on drift."
    ),
) -> None:
    """Print how to cite tulip, or check the citation metadata for version drift."""
    from tulip.cli._cite import check_version_parity, render_citation

    if check:
        drift = check_version_parity()
        if drift:
            _errors.print("citation metadata is out of sync:")
            for message in drift:
                _errors.print(f"  - {message}")
            raise typer.Exit(code=1)
        _console.print("[green]citation metadata versions agree[/green]")
        return

    _console.print(render_citation(citation_format))


# ------------------------------------------------------------------ helpers


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_frame(frame: Any, title: str) -> None:
    """Render a comparison DataFrame as a rich table under ``title``."""
    table = Table(title=title)
    for column in frame.columns:
        table.add_column(str(column))
    for _, row in frame.iterrows():
        table.add_row(*(_format_cell(value) for value in row))
    _console.print(table)


def main() -> None:
    """Console entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
