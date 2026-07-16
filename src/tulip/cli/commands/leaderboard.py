"""Leaderboard, efficiency, and reproducibility commands."""

from __future__ import annotations

from pathlib import Path

import typer

from tulip.cli._context import _console, _errors, _print_frame, _tulip_errors, app, repro_app


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
    _report_repro_drift(drift)


@repro_app.command("from-scratch")
@_tulip_errors
def repro_from_scratch(
    suite_path: Path = typer.Argument(..., help="Leaderboard suite YAML to reproduce."),
    against: Path = typer.Option(
        Path("benchmarks/results"), "--against", help="Committed artifact root to compare with."
    ),
) -> None:
    """Reproduce the committed board in full isolation, then match it byte-for-byte.

    Stronger than ``repro verify``: every build artifact (splits, trained models)
    is redirected into a throwaway directory too, so the run reads and writes
    nothing outside it. This proves the committed board depends only on the
    committed source, exactly as a fresh checkout in a clean container does, and
    is the command the Dockerfile runs. Byte-exact matching is platform sensitive,
    so run it on the platform that produced the committed board.
    """
    import tempfile

    from tulip.cli._repro import reproduce_from_scratch

    with tempfile.TemporaryDirectory(prefix="tulip-scratch-") as scratch:
        drift = reproduce_from_scratch(suite_path, against, Path(scratch))
    _report_repro_drift(drift)


def _report_repro_drift(drift: list[str]) -> None:
    """Print a reproduction result and exit non-zero on any drift."""
    if drift:
        _errors.print(f"reproduction FAILED: {len(drift)} artifact(s) drifted:")
        for line in drift:
            _errors.print(f"  - {line}")
        raise typer.Exit(code=1)
    _console.print("[green]reproduced the committed leaderboard byte-for-byte[/green]")
