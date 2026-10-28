"""Which tool calls run automatically, which need the user's approval, and which are blocked."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import Tool, ToolCall


class Risk(str, Enum):
    """What a tool can do. Drives approvals and whether calls may run in parallel."""

    READ = "read"  # inspects local state only
    WRITE = "write"  # changes files in the workspace
    NETWORK = "network"  # talks to other machines (read-only requests)
    EXEC = "exec"  # runs commands or changes remote state: anything can happen


class Approval(Enum):
    """An approver's answer for one tool call."""

    ALLOW = "allow"
    DENY = "deny"
    ALWAYS = "always"  # allow, and stop asking about this tool for the rest of the session


Approver = Callable[["Tool", "ToolCall"], Approval]

MODES = ("ask", "auto", "readonly")


class PermissionPolicy:
    """Decides whether a tool call may run.

    Modes:
      * ``ask`` (default): read-only tools run freely; anything else needs approval
        from ``approver`` unless listed in ``allow``.
      * ``auto``: everything runs (except what ``deny`` blocks).
      * ``readonly``: only read-only tools (plus anything in ``allow``) are offered to the model.

    ``allow`` and ``deny`` take tool names and/or risk levels (``"write"``, ``"network"``,
    ``"exec"``). ``deny`` always wins; denied tools aren't shown to the model at all.
    """

    def __init__(
        self,
        mode: str = "ask",
        *,
        allow: Iterable[str] = (),
        deny: Iterable[str] = (),
        approver: Approver | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown permission mode {mode!r}; choose from {', '.join(MODES)}")
        self.mode = mode
        self.allow = set(allow)
        self.deny = set(deny)
        self.approver = approver
        self._always: set[str] = set()

    def exposes(self, tool: Tool) -> bool:
        """Whether the model is told about this tool at all."""
        if _matches(self.deny, tool):
            return False
        if self.mode == "readonly":
            return tool.risk is Risk.READ or _matches(self.allow, tool)
        return True

    def requires_approval(self, tool: Tool) -> bool:
        return not (
            _matches(self.deny, tool)
            or tool.risk is Risk.READ
            or self.mode == "auto"
            or _matches(self.allow, tool)
            or tool.name in self._always
        )

    def check(self, tool: Tool, call: ToolCall) -> tuple[bool, str]:
        """Return ``(allowed, reason)``; ``reason`` explains a refusal."""
        if _matches(self.deny, tool):
            return False, f"{tool.name} is blocked by the permission policy"
        if not self.requires_approval(tool):
            return True, ""
        if self.mode == "readonly":
            return False, f"{tool.name} isn't available in read-only mode"
        if self.approver is None:
            return False, (
                f"{tool.name} ({tool.risk.value}) needs the user's approval and nobody is "
                f"available to approve it; the user can rerun with --yes or --allow {tool.name}"
            )
        answer = self.approver(tool, call)
        if answer is Approval.ALWAYS:
            self._always.add(tool.name)
            return True, ""
        if answer is Approval.ALLOW:
            return True, ""
        return False, "the user declined this tool call"


def _matches(names: set[str], tool: Tool) -> bool:
    return tool.name in names or tool.risk.value in names
