"""The tulip command-line interface.

One operator surface over the whole toolkit: dataset inspection and preparation,
training, benchmarking, evaluation, single-sample prediction (with optional map
export and explanations), and the HTTP service.

The root app, the shared consoles, the error boundary, and the render helpers
live in :mod:`tulip.cli._context`. Each command group lives in its own module
under :mod:`tulip.cli.commands`. This module assembles them by importing each
group for its registration side effects, and exposes the console entry point.
"""

from __future__ import annotations

from tulip.cli._context import app
from tulip.cli.commands import (  # noqa: F401  (imported for command registration)
    analyze,
    cards,
    data,
    inspect,
    leaderboard,
    pipeline,
    predict,
    registry,
    serve,
    training,
)

__all__ = ["app", "main"]


def main() -> None:
    """Console entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
