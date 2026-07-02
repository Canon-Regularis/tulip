"""Generic named-component registry.

Registries are how tulip stays modular: datasets, feature extractors, models,
and explainers are all registered under canonical string names, and experiment
configs refer to components purely by name. Adding a new model or feature is a
matter of writing one module and registering it -- no core code changes.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Generic, TypeVar

from tulip.core.exceptions import DuplicateComponentError, UnknownComponentError

T = TypeVar("T")


class Registry(Generic[T]):
    """A mapping from canonical names (and aliases) to components.

    Components are typically classes or zero-argument-friendly factories;
    :meth:`create` instantiates them with keyword arguments taken from config.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}
        self._aliases: dict[str, str] = {}

    @property
    def kind(self) -> str:
        """Human-readable description of what this registry holds (e.g. ``"model"``)."""
        return self._kind

    def register(self, name: str, *, aliases: Sequence[str] = ()) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``name``."""

        def decorator(obj: T) -> T:
            self.add(name, obj, aliases=aliases)
            return obj

        return decorator

    def add(self, name: str, obj: T, *, aliases: Sequence[str] = ()) -> None:
        """Register ``obj`` under ``name`` (and any ``aliases``)."""
        canonical = self._normalise(name)
        if canonical in self._items or canonical in self._aliases:
            raise DuplicateComponentError(f"{self._kind} {canonical!r} is already registered")
        self._items[canonical] = obj
        for alias in aliases:
            alias_key = self._normalise(alias)
            if alias_key in self._items or alias_key in self._aliases:
                raise DuplicateComponentError(
                    f"{self._kind} alias {alias_key!r} is already registered"
                )
            self._aliases[alias_key] = canonical

    def get(self, name: str) -> T:
        """Look up a component by canonical name or alias."""
        key = self._normalise(name)
        key = self._aliases.get(key, key)
        try:
            return self._items[key]
        except KeyError:
            suggestions = difflib.get_close_matches(key, self.names(), n=3)
            raise UnknownComponentError(self._kind, name, suggestions) from None

    def create(self, name: str, /, **kwargs: Any) -> Any:
        """Instantiate the component registered under ``name`` with ``kwargs``."""
        component = self.get(name)
        if not callable(component):
            raise TypeError(f"{self._kind} {name!r} is not callable and cannot be instantiated")
        return component(**kwargs)

    def names(self) -> list[str]:
        """Return all canonical component names, sorted."""
        return sorted(self._items)

    def items(self) -> list[tuple[str, T]]:
        """Return ``(name, component)`` pairs, sorted by name."""
        return sorted(self._items.items())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        key = self._normalise(name)
        return key in self._items or key in self._aliases

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry(kind={self._kind!r}, names={self.names()!r})"

    @staticmethod
    def _normalise(name: str) -> str:
        return name.strip().lower().replace("-", "_")
