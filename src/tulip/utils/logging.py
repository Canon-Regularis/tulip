"""Logging configuration for tulip.

Library modules obtain loggers via :func:`get_logger` and never configure
handlers themselves; only entry points (CLI, serving app) call
:func:`configure_logging`.
"""

from __future__ import annotations

import logging

_ROOT_LOGGER_NAME = "tulip"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``tulip`` namespace."""
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a rich console handler to the tulip root logger (entry points only)."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    if logger.handlers:  # already configured; avoid duplicate output
        return
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    except ImportError:  # rich is a hard dependency, but degrade gracefully anyway
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
