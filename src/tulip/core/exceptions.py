"""Exception hierarchy for tulip.

All exceptions raised by tulip code derive from :class:`TulipError` so callers
can catch toolkit failures with a single ``except`` clause.
"""

from __future__ import annotations


class TulipError(Exception):
    """Base class for all tulip errors."""


class ConfigurationError(TulipError):
    """An experiment or component configuration is invalid."""


class DataError(TulipError):
    """A dataset is missing, malformed, or fails validation."""


class UnknownComponentError(TulipError, KeyError):
    """A name was looked up in a registry that has no such component."""

    def __init__(self, kind: str, name: str, suggestions: list[str] | None = None) -> None:
        message = f"Unknown {kind}: {name!r}."
        if suggestions:
            message += f" Did you mean: {', '.join(repr(s) for s in suggestions)}?"
        super().__init__(message)
        self.kind = kind
        self.name = name
        self.suggestions = suggestions or []

    def __str__(self) -> str:  # KeyError.__str__ would repr() the message
        return self.args[0]


class DuplicateComponentError(TulipError):
    """A component name was registered twice in the same registry."""


class MissingDependencyError(TulipError, ImportError):
    """An optional dependency required for the requested feature is not installed."""

    def __init__(
        self, module: str, *, extra: str | None = None, purpose: str | None = None
    ) -> None:
        message = f"The optional dependency {module!r} is not installed"
        if purpose:
            message += f" (required for {purpose})"
        message += "."
        if extra:
            message += f' Install it with: pip install "tulip[{extra}]"'
        super().__init__(message)
        self.module = module
        self.extra = extra
