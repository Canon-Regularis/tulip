"""Shared CLI context: the Typer apps, consoles, error boundary, and render helpers.

The command modules in :mod:`tulip.cli.commands` import from here so every group
attaches to the same app tree and renders through the same sinks. Keeping this
separate from ``app.py`` is what lets each command group live in its own module
instead of one god-module.

Heavy imports stay inside command bodies so ``tulip --help`` never pays for
scikit-learn.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from tulip import __version__
from tulip.core.exceptions import ConfigurationError, TulipError
from tulip.utils.logging import configure_logging

if TYPE_CHECKING:
    from tulip.labels.taxonomy import LabelLevel

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

_CommandT = TypeVar("_CommandT", bound=Callable[..., None])


@dataclass
class _State:
    """Global toggles set by the root callback and read by the error boundary."""

    verbose: bool = False


_state = _State()


def _tulip_errors(command: _CommandT) -> _CommandT:
    """Decorate a command with the uniform TulipError boundary."""

    @functools.wraps(command)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            command(*args, **kwargs)
        except TulipError as exc:
            if _state.verbose:
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
    _state.verbose = verbose
    configure_logging(logging.DEBUG if verbose else logging.WARNING)


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


_NumberT = TypeVar("_NumberT", int, float)


def _parse_number_csv(
    value: str, cast: Callable[[str], _NumberT], *, name: str
) -> tuple[_NumberT, ...]:
    """Parse a comma-separated numeric CLI option, or raise a clean error.

    Empty tokens are ignored, so trailing commas and stray spaces are tolerated.
    A token ``cast`` rejects raises :class:`ConfigurationError` naming the option,
    rather than leaking a raw ``ValueError`` past the CLI error boundary.

    Args:
        value: The raw option string, e.g. ``"0.1,0.5,1.0"``.
        cast: Per-token parser, ``int`` or ``float``.
        name: The option name without dashes, e.g. ``"fractions"``; it names the
            option in the error message.

    Returns:
        The parsed numbers, in input order.

    Raises:
        ConfigurationError: if any non-empty token fails to parse.
    """
    try:
        return tuple(cast(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"--{name} must be comma-separated numbers, got {value!r}"
        ) from exc


def _parse_level(value: str) -> LabelLevel:
    """Parse a ``--level`` option into a :class:`LabelLevel`, or raise a clean error.

    A value outside the taxonomy raises :class:`ConfigurationError` listing the
    valid levels, rather than leaking a raw ``ValueError`` past the CLI error
    boundary.
    """
    from tulip.labels.taxonomy import LabelLevel

    try:
        return LabelLevel(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in LabelLevel)
        raise ConfigurationError(f"unknown level {value!r}; use one of: {allowed}") from exc


def _emit_report(
    report: Any,
    *,
    json_output: bool,
    out: Path | None = None,
    saved_label: str | None = None,
    markdown: str | None = None,
) -> None:
    """Save an optional report file, then print it as JSON or markdown.

    The shared tail of the pipeline report commands: when ``out`` is set, save the
    report there and note it; then print the report as JSON (``--json``) or as its
    rendered markdown.

    Args:
        report: A report exposing ``save``, ``model_dump_json``, and
            ``to_markdown``.
        json_output: When ``True``, print JSON instead of markdown.
        out: When set, ``report.save(out)`` runs and a confirmation prints.
        saved_label: The noun in the "written to" confirmation; used only when
            ``out`` is set.
        markdown: Pre-rendered markdown to print instead of
            ``report.to_markdown()`` (e.g. a ``to_markdown(top_k=...)`` variant).
    """
    if out is not None:
        report.save(out)
        _console.print(f"[green]{saved_label} written to {out}[/green]")
    if json_output:
        _console.print_json(report.model_dump_json())
    else:
        _console.print(markdown if markdown is not None else report.to_markdown())


def _read_json_mapping(path: Path, *, what: str) -> dict[str, Any]:
    """Read a JSON object, converting IO/parse failures into a clean ``DataError``.

    ``tulip.utils.io.read_json`` raises the underlying ``FileNotFoundError`` /
    ``JSONDecodeError``, and those escape the ``_tulip_errors`` boundary as a
    traceback that leaks absolute paths. Every other file-reading command in this
    CLI already fails with one clean ``error:`` line; this makes the card
    commands behave the same.
    """
    from tulip._jsonio import read_json_object
    from tulip.core.exceptions import DataError

    if not path.is_file():
        raise DataError(f"{what} not found: {path}")
    return read_json_object(path, what=what)
