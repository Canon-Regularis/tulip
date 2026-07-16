"""Corpus commands: catalog, download, prepare, synthesize, validate."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from tulip.cli._context import _console, _tulip_errors, datasets_app


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
