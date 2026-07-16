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
from typing import Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from tulip import __version__
from tulip.core.exceptions import TulipError
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
