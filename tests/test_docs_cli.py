"""Guard against drift between the documented CLI and the real one.

Two runbook commands once documented flags the CLI did not accept (`data prepare
--out`, `analyze --out`), so a copy-pasted step failed. This walks every `tulip`
invocation in the docs and asserts it still parses against the live command tree:
the subcommand exists and every ``--flag`` is real. It checks parsing, not
execution, so it is fast and needs no data.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import typer

from tulip.cli.app import app

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _cli_tree() -> dict[str, set[str]]:
    """Map every subcommand path (e.g. ``"data prepare"``) to its valid ``--flags``."""
    root = typer.main.get_command(app)
    tree: dict[str, set[str]] = {}

    def flags_of(command: object) -> set[str]:
        found: set[str] = set()
        for param in getattr(command, "params", []):
            names = list(getattr(param, "opts", [])) + list(getattr(param, "secondary_opts", []))
            found.update(name for name in names if name.startswith("--"))
        return found

    def walk(command: object, prefix: list[str]) -> None:
        subcommands = getattr(command, "commands", None)
        if subcommands:
            for name, sub in subcommands.items():
                walk(sub, [*prefix, name])
        else:
            tree[" ".join(prefix)] = flags_of(command)

    walk(root, [])
    return tree


def _documented_commands() -> list[tuple[str, str]]:
    """Return ``(source, command_line)`` for every tulip invocation in a code block."""
    commands: list[tuple[str, str]] = []
    docs = [*sorted((_REPO_ROOT / "docs").rglob("*.md")), _REPO_ROOT / "README.md"]
    for doc in docs:
        if not doc.is_file():
            continue
        in_block = False
        buffer = ""
        for raw in doc.read_text(encoding="utf-8").splitlines():
            if raw.lstrip().startswith("```"):
                in_block = not in_block
                continue
            if not in_block:
                continue
            stripped = raw.strip()
            if buffer:
                buffer += " " + stripped
            elif stripped.startswith(("tulip ", "$ tulip ")):
                buffer = stripped.removeprefix("$ ").strip()
            else:
                continue
            if buffer.endswith("\\"):  # shell line continuation
                buffer = buffer[:-1].strip()
            else:
                commands.append((str(doc.relative_to(_REPO_ROOT)), buffer))
                buffer = ""
        if buffer:
            commands.append((str(doc.relative_to(_REPO_ROOT)), buffer))
    return commands


_DOCUMENTED = _documented_commands()


def test_docs_reference_some_tulip_commands() -> None:
    # If extraction silently finds nothing, the parse test below is vacuous.
    assert len(_DOCUMENTED) >= 20


@pytest.mark.parametrize(
    ("source", "command_line"),
    _DOCUMENTED,
    ids=[f"{src}:{line[:50]}" for src, line in _DOCUMENTED],
)
def test_documented_command_parses(source: str, command_line: str) -> None:
    tree = _cli_tree()
    # Trim at the first shell operator, redirect, or comment; keep the tulip call.
    head = re.split(r"[|><#]| && | ; ", command_line)[0].strip()
    tokens = shlex.split(head)
    assert tokens and tokens[0] == "tulip", command_line
    rest = tokens[1:]

    two, one = " ".join(rest[:2]), (rest[0] if rest else "")
    if two in tree:
        command, flag_start = two, 2
    elif one in tree:
        command, flag_start = one, 1
    else:
        pytest.fail(f"{source}: no such command in `{command_line}` (have: {sorted(tree)[:5]}...)")

    valid = tree[command]
    for token in rest[flag_start:]:
        if token.startswith("--"):
            flag = token.split("=", 1)[0]
            assert flag in valid, (
                f"{source}: `tulip {command}` has no {flag!r} in `{command_line}` "
                f"(valid: {sorted(valid)})"
            )
