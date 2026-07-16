"""Discovery commands: doctor, registry listings, and citation."""

from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from tulip.cli._context import (
    _console,
    _errors,
    _tulip_errors,
    app,
    explainers_app,
    features_app,
    models_app,
)


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
