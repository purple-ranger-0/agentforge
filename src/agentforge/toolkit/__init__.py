"""Built-in tool packs.

    from agentforge import Agent
    from agentforge.toolkit import load

    agent = Agent(load(["fs", "git"]))      # packs and/or individual tool names

Packs: ``fs`` (files), ``git``, ``shell`` (run_command), ``net`` (HTTP, DNS, ports, TLS),
``system`` (machine, processes, listening ports) and ``data`` (JWT, encodings, hashes,
time, cron, regex, calculator, CIDR, secrets, JSON).
"""

from __future__ import annotations

from collections.abc import Iterable

from ..tools import Tool
from . import data, fs, git, net, shell, system

PACKS: dict[str, tuple[Tool, ...]] = {
    "fs": fs.TOOLS,
    "git": git.TOOLS,
    "shell": shell.TOOLS,
    "net": net.TOOLS,
    "system": system.TOOLS,
    "data": data.TOOLS,
}

ALL_TOOLS: dict[str, Tool] = {t.name: t for tools in PACKS.values() for t in tools}


def pack_of(tool: Tool) -> str | None:
    """The built-in pack a tool belongs to, if any."""
    for name, tools in PACKS.items():
        if tool in tools:
            return name
    return None


def load(names: Iterable[str] = ("all",)) -> list[Tool]:
    """Resolve pack names and/or tool names (or "all") into a list of tools, in order."""
    selected: dict[str, Tool] = {}
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name == "all":
            selected.update(ALL_TOOLS)
        elif name in PACKS:
            selected.update((t.name, t) for t in PACKS[name])
        elif name in ALL_TOOLS:
            selected[name] = ALL_TOOLS[name]
        else:
            choices = ", ".join([*PACKS, "all"])
            raise ValueError(f"unknown tool pack or tool {name!r}; packs are: {choices}")
    return list(selected.values())


__all__ = ["ALL_TOOLS", "PACKS", "load", "pack_of"]
