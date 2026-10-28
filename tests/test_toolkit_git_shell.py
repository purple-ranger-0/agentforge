from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agentforge.errors import ToolError
from agentforge.toolkit import git, shell

Run = Callable[..., Any]


@pytest.fixture
def repo(tmp_path: Path, git_repo: Callable[..., str]) -> Callable[..., str]:
    (tmp_path / "app.py").write_text("def main():\n    return 1\n")
    git_repo("add", ".")
    git_repo("commit", "-q", "-m", "Add app")
    (tmp_path / "app.py").write_text("def main():\n    return 2\n")
    (tmp_path / "notes.md").write_text("# notes\n")
    git_repo("add", "notes.md")
    git_repo("commit", "-q", "-m", "Add notes\n\nWith a body.")
    git_repo("tag", "v1.0")
    return git_repo


def test_git_status(repo: Callable[..., str], run: Run, tmp_path: Path) -> None:
    (tmp_path / "new.txt").write_text("")
    out = run(git.git_status)
    lines = out.splitlines()
    assert lines[0] == "## main"
    assert lines[1].startswith("Last commit: ") and "Add notes" in lines[1]
    assert " M app.py" in lines
    assert "?? new.txt" in lines


def test_git_status_clean_and_empty_repo(git_repo: Callable[..., str], run: Run) -> None:
    out = run(git.git_status)
    assert "(no commits yet)" in out
    assert out.endswith("Working tree clean.")


def test_git_log(repo: Callable[..., str], run: Run) -> None:
    out = run(git.git_log)
    first, second = out.splitlines()
    assert first.endswith("Ada Lovelace: Add notes (HEAD -> main, tag: v1.0)")
    assert second.endswith("Ada Lovelace: Add app")
    assert run(git.git_log, path="notes.md").count("\n") == 0
    assert "Add app" in run(git.git_log, grep="ADD APP")
    assert run(git.git_log, author="nobody") == "No matching commits."
    assert "notes.md" in run(git.git_log, max_count=1, stat=True)
    with pytest.raises(ToolError, match="invalid ref"):
        run(git.git_log, ref="--output=/tmp/pwned")


def test_git_diff(repo: Callable[..., str], run: Run, tmp_path: Path) -> None:
    out = run(git.git_diff)
    assert "-    return 1" in out and "+    return 2" in out
    assert run(git.git_diff, staged=True) == "No differences."
    assert "app.py" in run(git.git_diff, stat_only=True)
    assert "notes.md" in run(git.git_diff, target="HEAD~1", stat_only=True)
    with pytest.raises(ToolError, match="invalid target"):
        run(git.git_diff, target="-p")


def test_git_diff_reports_no_changes(repo: Callable[..., str], run: Run) -> None:
    repo("checkout", "-q", "app.py")
    assert run(git.git_diff).startswith("No unstaged changes.")


def test_git_show(repo: Callable[..., str], run: Run) -> None:
    commit = run(git.git_show)
    assert "Add notes" in commit and "With a body." in commit and "+# notes" in commit
    assert "notes.md | 1 +" in run(git.git_show, mode="stat")
    assert (
        run(git.git_show, ref="HEAD~1", path="app.py", mode="file") == "def main():\n    return 1\n"
    )
    with pytest.raises(ToolError, match="needs a path"):
        run(git.git_show, mode="file")


def test_git_blame(repo: Callable[..., str], run: Run) -> None:
    out = run(git.git_blame, path="app.py")
    assert "Ada Lovelace" in out and "def main" in out
    assert len(run(git.git_blame, path="app.py", start_line=2, end_line=2).splitlines()) == 1
    assert "return 1" in run(git.git_blame, path="app.py", ref="HEAD")
    with pytest.raises(ToolError, match="has only 2 lines"):
        run(git.git_blame, path="app.py", start_line=10)
    with pytest.raises(ToolError, match="end_line"):
        run(git.git_blame, path="app.py", start_line=2, end_line=1)


def test_git_branches(repo: Callable[..., str], run: Run) -> None:
    repo("branch", "feature")
    lines = run(git.git_branches).splitlines()
    assert lines[0].startswith("* main") or lines[1].startswith("* main")
    assert any(" feature " in f" {line} " for line in lines)


def test_git_outside_a_repository(run: Run) -> None:
    with pytest.raises(ToolError, match="isn't inside a git repository"):
        run(git.git_log)


# ------------------------------------------------------------------------------ shell


def test_run_command_reports_exit_code_and_output(run: Run, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    out = run(shell.run_command, command="echo hi; echo oops >&2; pwd; exit 3", cwd="sub")
    status, *lines = out.splitlines()
    assert status.startswith("[exit code 3, ")
    assert lines == ["hi", "oops", str(tmp_path / "sub")]


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell")
def test_run_command_timeout_kills_the_process_group(run: Run) -> None:
    out = run(shell.run_command, command="sleep 30 & sleep 30", timeout=1)
    assert out.startswith("[killed after the 1s timeout")


def test_run_command_environment_and_ansi(run: Run) -> None:
    out = run(
        shell.run_command, command="printf '\\033[31mred\\033[0m'; echo; echo $NO_COLOR $PAGER"
    )
    assert out.splitlines()[1:] == ["red", "1 cat"]


def test_run_command_keeps_head_and_tail_of_huge_output(run: Run) -> None:
    command = f"{sys.executable} -c \"print('a' * 150000 + 'MIDDLE' + 'z' * 150000)\""
    out = run(shell.run_command, command=command)
    assert "bytes of output omitted" in out
    assert "MIDDLE" not in out
    assert out.rstrip().endswith("z" * 100)


def test_run_command_validation(run: Run) -> None:
    with pytest.raises(ToolError, match="empty"):
        run(shell.run_command, command="  ")
    with pytest.raises(ToolError, match="outside the workspace"):
        run(shell.run_command, command="ls", cwd="/")
