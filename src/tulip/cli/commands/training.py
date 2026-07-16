"""Training and benchmarking commands."""

from __future__ import annotations

from pathlib import Path

import typer

from tulip.cli._context import _console, _print_frame, _tulip_errors, app


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
