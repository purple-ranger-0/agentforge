from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agentforge import Tool, ToolContext, Workspace


@pytest.fixture(autouse=True)
def _never_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never ask for approval on the developer's real terminal."""
    monkeypatch.setattr("agentforge.render.Terminal.available", lambda self: False)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path)


@pytest.fixture
def ctx(ws: Workspace) -> ToolContext:
    return ToolContext(workspace=ws)


@pytest.fixture
def run(ctx: ToolContext) -> Callable[..., Any]:
    """Validate arguments the way the agent does, then call the tool."""

    def call(tool: Tool, **args: Any) -> Any:
        return tool.invoke(tool.validate(args), ctx)

    return call


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    """An isolated git repository in tmp_path; returns a function that runs git in it."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    home = tmp_path.parent / f"{tmp_path.name}-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{var}_NAME", "Ada Lovelace")
        monkeypatch.setenv(f"GIT_{var}_EMAIL", "ada@example.com")

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout

    git("init", "-q")
    return git
