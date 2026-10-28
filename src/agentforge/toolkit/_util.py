"""Helpers shared by the built-in tools."""

from __future__ import annotations

import datetime as dt
import difflib
import functools
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

# Directories that are almost never what a search is looking for.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".next",
        ".gradle",
        ".terraform",
        ".idea",
    }
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def is_binary(path: Path, sample: int = 8192) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(sample)
    except OSError:
        return False


def local_iso(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def relative_time(when: dt.datetime, now: dt.datetime | None = None) -> str:
    """'in 3 days, 4 hours' / '12 minutes ago'."""
    now = now or dt.datetime.now(dt.timezone.utc)
    seconds = (when - now).total_seconds()
    if abs(seconds) < 1:
        return "now"
    span = format_duration(abs(seconds))
    return f"in {span}" if seconds > 0 else f"{span} ago"


def format_duration(seconds: float) -> str:
    """'3 days, 4 hours' (the two most significant units)."""
    units = (("year", 365 * 86400), ("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1))
    for index, (name, size) in enumerate(units):
        if seconds >= size:
            count = int(seconds // size)
            text = f"{count} {name}{'' if count == 1 else 's'}"
            if index + 1 < len(units):
                next_name, next_size = units[index + 1]
                rest = int((seconds - count * size) // next_size)
                if rest:
                    text += f", {rest} {next_name}{'' if rest == 1 else 's'}"
            return text
    return "0 seconds"


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        env=env,
    )


def unified_diff(
    old: str, new: str, old_name: str, new_name: str, *, context: int = 3, max_lines: int = 200
) -> str:
    lines = [
        line.rstrip("\n")
        for line in difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            old_name,
            new_name,
            n=context,
        )
    ]
    if len(lines) > max_lines:
        lines = [*lines[:max_lines], f"... ({len(lines) - max_lines} more diff lines)"]
    return "\n".join(lines)


# ----------------------------------------------------------------- file discovery


def glob_match(relpath: str, pattern: str) -> bool:
    """Match a /-separated relative path against a glob.

    Supports ``*``, ``?``, ``[abc]``, ``{a,b}`` and ``**``. A pattern without ``/`` matches
    the file name at any depth (like ``rg -g``); otherwise it matches the whole path.
    """
    pattern = pattern.strip()
    while pattern.startswith("./"):
        pattern = pattern[2:]
    if "/" not in pattern:
        relpath = relpath.rsplit("/", 1)[-1]
    return _glob_regex(pattern).fullmatch(relpath) is not None


@functools.lru_cache(maxsize=256)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(_glob_to_regex(pattern), re.DOTALL)


def _glob_to_regex(pattern: str) -> str:
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            end = pattern.find("]", i + 2)
            if end == -1:
                out.append(re.escape(char))
            else:
                body = pattern[i + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = end + 1
                continue
        elif char == "{":
            end = pattern.find("}", i)
            if end == -1:
                out.append(re.escape(char))
            else:
                options = pattern[i + 1 : end].split(",")
                out.append("(?:" + "|".join(_glob_to_regex(o) for o in options) + ")")
                i = end + 1
                continue
        else:
            out.append(re.escape(char))
        i += 1
    return "".join(out)


def iter_files(start: Path) -> Iterator[Path]:
    """Files under ``start`` (or ``start`` itself if it's a file), in a stable order.

    Inside a git work tree this asks git, so .gitignored files are skipped; elsewhere it
    walks the tree, skipping :data:`IGNORED_DIRS`.
    """
    if start.is_file():
        yield start
        return
    listed = _git_listed_files(start)
    if listed is not None:
        yield from listed
        return
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath, name)


def _git_listed_files(start: Path) -> list[Path] | None:
    if shutil.which("git") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(start),
                "-c",
                "core.fsmonitor=false",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    names = dict.fromkeys(
        n for n in proc.stdout.decode("utf-8", "surrogateescape").split("\0") if n
    )
    return [start / name for name in sorted(names) if (start / name).is_file()]
