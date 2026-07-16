"""Model-registry commands: add, promote, rollback, ls, resolve."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from tulip.cli._context import _console, _read_json_mapping, _tulip_errors, registry_app

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
