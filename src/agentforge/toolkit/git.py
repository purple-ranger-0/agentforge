"""Read-only git tools, run against the repository that contains the workspace root."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Literal

from ..errors import ToolError
from ..tools import ToolContext, tool
from ._util import clamp, run

_STATUS_LEGEND = (
    "(XY codes: X = staged, Y = unstaged; M modified, A added, D deleted, R renamed, ?? untracked)"
)


def _git(ctx: ToolContext, *args: str, timeout: float = 30.0) -> str:
    if shutil.which("git") is None:
        raise ToolError("git isn't installed (not found on PATH)")
    cmd = [
        "git",
        "-C",
        str(ctx.workspace.root),
        "--no-pager",
        # Hardening: never run an fsmonitor hook, and keep output plain.
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.quotepath=false",
        "-c",
        "color.ui=never",
        *args,
    ]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "GIT_PAGER": "cat"}
    try:
        proc = run(cmd, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"git {args[0]} timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0:
        message = proc.stderr.strip() or f"git exited with status {proc.returncode}"
        if "not a git repository" in message:
            raise ToolError(f"{ctx.workspace.root} isn't inside a git repository")
        raise ToolError(message)
    return proc.stdout


def _rev(value: str, what: str = "revision") -> str:
    """Validate a revision so it can't be mistaken for a command-line option."""
    rev = value.strip()
    if not rev or rev.startswith("-") or any(c.isspace() or c == "\0" for c in rev):
        raise ToolError(f"invalid {what}: {value!r}")
    return rev


def _pathspec(ctx: ToolContext, path: str) -> str:
    target = ctx.workspace.resolve(path)  # may no longer exist in the working tree
    return os.path.relpath(target, ctx.workspace.root)


@tool
def git_status(ctx: ToolContext) -> str:
    """Show the current branch, its upstream (ahead/behind), the last commit, and staged,
    unstaged and untracked changes. Start here for questions about the state of a repository."""
    lines = _git(ctx, "status", "--short", "--branch", "--untracked-files=normal").splitlines()
    branch, changes = (lines[0] if lines else ""), lines[1:]
    try:
        last = _git(ctx, "log", "-1", "--format=%h %s (%cr, %an)").strip()
    except ToolError:
        last = "(no commits yet)"
    parts = [branch, f"Last commit: {last}"]
    if changes:
        parts.append(f"{len(changes)} changed path(s) {_STATUS_LEGEND}:")
        parts.extend(changes)
    else:
        parts.append("Working tree clean.")
    stashes = _git(ctx, "stash", "list").splitlines()
    if stashes:
        parts.append(f"{len(stashes)} stash entr{'y' if len(stashes) == 1 else 'ies'}.")
    return "\n".join(parts)


@tool
def git_log(
    ctx: ToolContext,
    max_count: int = 20,
    path: str | None = None,
    ref: str | None = None,
    author: str | None = None,
    grep: str | None = None,
    since: str | None = None,
    stat: bool = False,
) -> str:
    """Show commit history, newest first: hash, date, author, subject and branch/tag names.

    Use it to see what changed recently, find when something was introduced, or summarize
    work. Combine filters to narrow down.

    Args:
        max_count: Maximum number of commits (1-500).
        path: Only commits touching this file or directory (follows renames for a file).
        ref: Branch, tag or revision range to list, e.g. "main", "v1.2.0..HEAD". Default: HEAD.
        author: Only commits whose author matches this text (name or email).
        grep: Only commits whose message matches this regex (case-insensitive).
        since: Only commits newer than this, e.g. "2 weeks ago" or "2026-01-31".
        stat: Also list the files each commit changed.
    """
    args = [
        "log",
        "--no-ext-diff",
        "--no-textconv",
        f"--max-count={clamp(max_count, 1, 500)}",
        "--date=short",
        "--format=%h %ad %an: %s%d",
    ]
    if author:
        args.append(f"--author={author}")
    if grep:
        args += [f"--grep={grep}", "--regexp-ignore-case"]
    if since:
        args.append(f"--since={since}")
    if stat:
        args.append("--stat")
    pathspec = _pathspec(ctx, path) if path else None
    if pathspec and (ctx.workspace.root / pathspec).is_file():
        args.append("--follow")
    if ref:
        args.append(_rev(ref, "ref"))
    if pathspec:
        args += ["--", pathspec]
    output = _git(ctx, *args)
    return output.rstrip() or "No matching commits."


@tool
def git_diff(
    ctx: ToolContext,
    target: str | None = None,
    staged: bool = False,
    path: str | None = None,
    stat_only: bool = False,
    context_lines: int = 3,
) -> str:
    """Show changes as a unified diff.

    By default shows unstaged changes in the working tree. staged=true shows what will go
    into the next commit. `target` compares against a commit or branch ("main", "HEAD~3"),
    and "main...feature" shows what a branch changed since it forked. Use it to review
    work, draft commit messages, or find what broke.

    Args:
        target: Commit, branch, or range to compare against.
        staged: Show staged changes instead of unstaged ones.
        path: Limit the diff to this file or directory.
        stat_only: Only summarize changed files and line counts.
        context_lines: Unchanged lines to show around each change (0-20).
    """
    args = [
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--find-renames",
        f"--unified={clamp(context_lines, 0, 20)}",
    ]
    if staged:
        args.append("--cached")
    if stat_only:
        args.append("--stat")
    if target:
        args.append(_rev(target, "target"))
    args.append("--")
    if path:
        args.append(_pathspec(ctx, path))
    output = _git(ctx, *args)
    if output.strip():
        return output.rstrip()
    if not staged and not target:
        return (
            "No unstaged changes. (Use staged=true for staged changes, or a target to "
            "compare commits.)"
        )
    return "No differences."


@tool
def git_show(
    ctx: ToolContext,
    ref: str = "HEAD",
    path: str | None = None,
    mode: Literal["commit", "stat", "file"] = "commit",
) -> str:
    """Show a commit's message and diff, or a file as it was at some revision.

    mode="commit" shows the full commit, mode="stat" only the list of changed files, and
    mode="file" prints `path` as it existed at `ref` (e.g. ref="HEAD~5" or "v1.0.0").

    Args:
        ref: Commit hash, branch, tag or relative ref like HEAD~2.
        path: Limit the commit to this path, or the file to print in mode="file".
        mode: What to show.
    """
    rev = _rev(ref, "ref")
    if mode == "file":
        if not path:
            raise ToolError("mode='file' needs a path")
        return _git(ctx, "show", f"{rev}:./{_pathspec(ctx, path)}")
    args = ["show", "--no-ext-diff", "--no-textconv", "--find-renames", "--format=fuller"]
    if mode == "stat":
        args.append("--stat")
    args.append(rev)
    if path:
        args += ["--", _pathspec(ctx, path)]
    return _git(ctx, *args).rstrip()


@tool
def git_blame(
    ctx: ToolContext,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    ref: str | None = None,
) -> str:
    """Show who last changed each line of a file, in which commit and when.

    Use it to find when and why a line was introduced; follow up with git_show on the commit.
    At most 400 lines per call.

    Args:
        path: File to blame.
        start_line: First line (1-based).
        end_line: Last line (defaults to start_line + 199).
        ref: Blame the file as of this revision instead of the working tree.
    """
    start = max(1, start_line)
    end = end_line if end_line is not None else start + 199
    if end < start:
        raise ToolError("end_line must be >= start_line")
    end = min(end, start + 399)
    if not ref:
        file = ctx.workspace.resolve(path, kind="file")
        with file.open("rb") as fh:
            line_count = sum(1 for _ in fh)
        if start > line_count:
            raise ToolError(f"{path} has only {line_count} lines")
        end = min(end, line_count)
    args = ["blame", "--no-textconv", "--date=short", f"-L{start},{end}"]
    if ref:
        args.append(_rev(ref, "ref"))
    args += ["--", _pathspec(ctx, path)]
    return _git(ctx, *args).rstrip()


@tool
def git_branches(ctx: ToolContext, include_remote: bool = False, max_count: int = 30) -> str:
    """List branches, most recently committed first, with last commit date, author, upstream
    tracking state (ahead/behind) and subject. The current branch is marked with *.

    Args:
        include_remote: Include remote-tracking branches too.
        max_count: Maximum number of branches (1-200).
    """
    refs = ["refs/heads"] + (["refs/remotes"] if include_remote else [])
    fmt = (
        "%(HEAD)\x1f%(refname:short)\x1f%(committerdate:short)\x1f%(authorname)"
        "\x1f%(upstream:short) %(upstream:track)\x1f%(subject)"
    )
    output = _git(
        ctx,
        "for-each-ref",
        "--sort=-committerdate",
        f"--count={clamp(max_count, 1, 200)}",
        f"--format={fmt}",
        *refs,
    )
    lines = []
    for row in output.splitlines():
        head, name, date, author, upstream, subject = row.split("\x1f", 5)
        tracking = f" [{upstream.strip()}]" if upstream.strip() else ""
        lines.append(f"{head or ' '} {name}{tracking}  {date}  {author}: {subject}")
    return "\n".join(lines) or "No branches (the repository has no commits yet)."


TOOLS = (git_status, git_log, git_diff, git_show, git_blame, git_branches)
