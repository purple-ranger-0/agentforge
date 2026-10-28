from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentforge import Approval, PermissionPolicy, Risk, ToolCall, Workspace, tool
from agentforge.errors import ToolError


@tool
def look() -> str:
    """Look."""
    return "ok"


@tool(risk="write")
def save() -> str:
    """Save."""
    return "ok"


@tool(risk="exec")
def shell() -> str:
    """Shell."""
    return "ok"


@tool(risk="network")
def ping() -> str:
    """Ping."""
    return "ok"


def call(name: str) -> ToolCall:
    return ToolCall(id="t1", name=name, input={})


def test_ask_mode_runs_reads_and_asks_for_the_rest() -> None:
    asked = []

    def approver(t, c):
        asked.append(t.name)
        return Approval.ALLOW

    policy = PermissionPolicy(approver=approver)
    assert policy.check(look, call("look")) == (True, "")
    assert policy.check(save, call("save")) == (True, "")
    assert asked == ["save"]


def test_deny_answer_and_missing_approver() -> None:
    policy = PermissionPolicy(approver=lambda t, c: Approval.DENY)
    assert policy.check(shell, call("shell")) == (False, "the user declined this tool call")
    allowed, reason = PermissionPolicy().check(shell, call("shell"))
    assert not allowed
    assert "--yes" in reason and "--allow shell" in reason


def test_always_stops_asking_for_that_tool() -> None:
    answers = iter([Approval.ALWAYS])
    policy = PermissionPolicy(approver=lambda t, c: next(answers))
    assert policy.check(save, call("save"))[0]
    assert not policy.requires_approval(save)
    assert policy.check(save, call("save"))[0]  # no second answer needed
    assert policy.requires_approval(shell)


def test_allow_by_name_or_risk() -> None:
    policy = PermissionPolicy(allow=["network", "save"])
    assert not policy.requires_approval(ping)
    assert not policy.requires_approval(save)
    assert policy.requires_approval(shell)


def test_auto_mode_and_deny_list() -> None:
    policy = PermissionPolicy("auto", deny=["exec"])
    assert policy.check(save, call("save")) == (True, "")
    assert policy.check(shell, call("shell"))[0] is False
    assert not policy.exposes(shell)
    assert policy.exposes(save)


def test_readonly_mode_hides_risky_tools() -> None:
    policy = PermissionPolicy("readonly", allow=["network"])
    assert policy.exposes(look)
    assert policy.exposes(ping)  # explicitly allowed
    assert not policy.exposes(save)
    assert not policy.check(save, call("save"))[0]


def test_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown permission mode"):
        PermissionPolicy("yolo")


def test_risk_values() -> None:
    assert [r.value for r in Risk] == ["read", "write", "network", "exec"]


# --------------------------------------------------------------------------- workspace


def test_resolve_relative_and_absolute(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.txt").write_text("x")
    ws = Workspace(tmp_path)
    assert ws.resolve("a/f.txt", kind="file") == tmp_path / "a" / "f.txt"
    assert ws.resolve(str(tmp_path / "a"), kind="dir") == tmp_path / "a"
    assert ws.resolve("new/file.txt") == tmp_path / "new" / "file.txt"
    assert ws.display(tmp_path / "a" / "f.txt") == "a/f.txt"
    assert ws.display(tmp_path) == "."


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "a/../../x"])
def test_paths_outside_the_root_are_refused(tmp_path: Path, path: str) -> None:
    (tmp_path / "a").mkdir()
    with pytest.raises(ToolError, match="outside the workspace root"):
        Workspace(tmp_path).resolve(path)


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinks_cannot_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("s3cret")
    (root / "link").symlink_to(secret)
    with pytest.raises(ToolError, match="outside the workspace root"):
        Workspace(root).resolve("link")


def test_kind_checks(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "f").write_text("")
    ws = Workspace(tmp_path)
    with pytest.raises(ToolError, match="no such file"):
        ws.resolve("missing", kind="file")
    with pytest.raises(ToolError, match="not a regular file"):
        ws.resolve("d", kind="file")
    with pytest.raises(ToolError, match="not a directory"):
        ws.resolve("f", kind="dir")
    with pytest.raises(ToolError, match="no such file or directory"):
        ws.resolve("missing", kind="exists")
    with pytest.raises(ToolError, match="empty"):
        ws.resolve("  ")


def test_root_must_be_a_directory(tmp_path: Path) -> None:
    (tmp_path / "f").write_text("")
    with pytest.raises(NotADirectoryError):
        Workspace(tmp_path / "f")
