"""File tools: read, list, find, search, inspect, compare, write and edit files in the workspace."""

from __future__ import annotations

import collections
import heapq
import mimetypes
import os
import re
import shutil
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..tools import ToolContext, tool
from ._util import (
    IGNORED_DIRS,
    clamp,
    glob_match,
    human_size,
    is_binary,
    iter_files,
    local_iso,
    unified_diff,
)

_MAX_LINE_CHARS = 2000
_COUNT_LINES_UP_TO = 20 * 1024 * 1024  # don't count lines of bigger files
_SEARCH_MAX_FILE = 10 * 1024 * 1024
_SEARCH_SECONDS = 20.0
_MAX_FILES_SCANNED = 200_000


def _clip(line: str, limit: int = _MAX_LINE_CHARS) -> str:
    if len(line) <= limit:
        return line
    return f"{line[:limit]}... [+{len(line) - limit} chars]"


@tool
def read_file(ctx: ToolContext, path: str, start_line: int = 1, max_lines: int = 400) -> str:
    """Read a text file with line numbers.

    Use this before answering questions about source code, configs or logs, and before
    editing a file. Page through big files with start_line/max_lines; a negative
    start_line counts from the end, e.g. -200 shows the last 200 lines of a log.

    Args:
        path: File path, relative to the workspace root.
        start_line: First line to show (1-based). Negative values count back from the end.
        max_lines: Maximum number of lines to return (1-2000).
    """
    file = ctx.workspace.resolve(path, kind="file")
    shown = ctx.workspace.display(file)
    max_lines = clamp(max_lines, 1, 2000)
    size = file.stat().st_size
    if is_binary(file):
        raise ToolError(
            f"{shown} looks like a binary file ({human_size(size)}); "
            "use file_info or compute_hash instead"
        )
    total: int | None = 0
    with file.open(encoding="utf-8", errors="replace") as fh:
        if start_line < 0:
            tail = collections.deque(enumerate(fh, 1), maxlen=-start_line)
            total = tail[-1][0] if tail else 0
            selected = list(tail)[:max_lines]
        else:
            first = max(start_line, 1)
            selected = []
            for total, line in enumerate(fh, 1):
                if total < first:
                    continue
                if len(selected) < max_lines:
                    selected.append((total, line))
                elif size > _COUNT_LINES_UP_TO:
                    total = None  # too big to count; we only know more lines follow
                    break
    if total == 0:
        return f"{shown} is empty."
    if not selected:
        return f"{shown} has {total} lines; start_line {start_line} is past the end."
    body = "\n".join(f"{n:>6}\t{_clip(line.rstrip(chr(10)))}" for n, line in selected)
    first_shown, last_shown = selected[0][0], selected[-1][0]
    header = f"{shown} (lines {first_shown}-{last_shown} of {total if total else '?'})"
    if total is None:
        footer = f"\n[more lines follow; continue with start_line={last_shown + 1}]"
    elif last_shown < total:
        footer = f"\n[{total - last_shown} more lines; continue with start_line={last_shown + 1}]"
    else:
        footer = ""
    return f"{header}\n{body}{footer}"


@tool
def list_dir(ctx: ToolContext, path: str = ".", depth: int = 2, show_hidden: bool = True) -> str:
    """List a directory as an indented tree, with file sizes.

    Use this to get oriented in a project before searching or reading files.
    Dependency and VCS folders (node_modules, .git, .venv, ...) are summarized, not expanded.

    Args:
        path: Directory to list, relative to the workspace root.
        depth: How many levels to expand (1-6).
        show_hidden: Include dotfiles and dot-directories.
    """
    base = ctx.workspace.resolve(path, kind="dir")
    depth = clamp(depth, 1, 6)
    lines = [f"{ctx.workspace.display(base)}/"]
    budget = 500

    def walk(directory: Path, level: int) -> bool:
        nonlocal budget
        try:
            entries = sorted(os.scandir(directory), key=lambda e: (not _is_dir(e), e.name.lower()))
        except OSError as exc:
            lines.append(f"{'  ' * (level + 1)}[unreadable: {exc.strerror}]")
            return True
        for entry in entries:
            if not show_hidden and entry.name.startswith("."):
                continue
            if budget <= 0:
                lines.append(
                    "[listing truncated at 500 entries; list a subdirectory or use a lower depth]"
                )
                return False
            budget -= 1
            indent = "  " * (level + 1)
            if entry.is_symlink():
                lines.append(f"{indent}{entry.name} -> {os.readlink(entry.path)}")
            elif _is_dir(entry):
                if entry.name in IGNORED_DIRS or level + 1 >= depth:
                    lines.append(f"{indent}{entry.name}/ ({_count_entries(entry.path)} entries)")
                else:
                    lines.append(f"{indent}{entry.name}/")
                    if not walk(Path(entry.path), level + 1):
                        return False
            else:
                try:
                    size = human_size(entry.stat().st_size)
                except OSError:
                    size = "?"
                lines.append(f"{indent}{entry.name}  ({size})")
        return True

    walk(base, 0)
    return "\n".join(lines)


def _is_dir(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_dir(follow_symlinks=False)
    except OSError:
        return False


def _count_entries(path: str) -> int | str:
    try:
        return len(os.listdir(path))
    except OSError:
        return "?"


@tool
def find_files(ctx: ToolContext, pattern: str, path: str = ".", max_results: int = 200) -> str:
    """Find files by name or path pattern (glob).

    Examples: "*.py", "Dockerfile*", "*.{yml,yaml}", "src/**/test_*.ts". A pattern without
    "/" matches file names at any depth; one with "/" matches the path relative to `path`.
    Respects .gitignore inside git repositories. Returns files only, not directories.

    Args:
        pattern: Glob pattern to match.
        path: Directory to search, relative to the workspace root.
        max_results: Maximum number of paths to return (1-2000).
    """
    base = ctx.workspace.resolve(path, kind="dir")
    max_results = clamp(max_results, 1, 2000)
    matches: list[str] = []
    total = scanned = 0
    for file in iter_files(base):
        scanned += 1
        if scanned > _MAX_FILES_SCANNED:
            break
        if glob_match(file.relative_to(base).as_posix(), pattern):
            total += 1
            if len(matches) < max_results:
                matches.append(ctx.workspace.display(file))
    where = ctx.workspace.display(base)
    if not matches:
        return f"No files matching {pattern!r} under {where}."
    header = f"{total} file(s) matching {pattern!r} under {where}"
    if total > len(matches):
        header += f" (showing the first {len(matches)})"
    if scanned > _MAX_FILES_SCANNED:
        header += f" (stopped after scanning {_MAX_FILES_SCANNED:,} files)"
    return header + ":\n" + "\n".join(matches)


@tool
def search_text(
    ctx: ToolContext,
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
    literal: bool = False,
    context_lines: int = 0,
    max_results: int = 100,
) -> str:
    """Search file contents with a regular expression, like `grep -rn`.

    Use it to find where something is defined, used, configured or logged. Respects
    .gitignore in git repositories and skips binary files. Output lines look like
    "path:line:text" (context lines use "-" instead of ":").

    Args:
        pattern: Regular expression (Python syntax), or plain text when literal is true.
        path: File or directory to search, relative to the workspace root.
        glob: Only search files matching this glob, e.g. "*.py" or "src/**/*.{ts,tsx}".
        ignore_case: Match case-insensitively.
        literal: Treat the pattern as plain text instead of a regex.
        context_lines: Lines of context to show around each match (0-10).
        max_results: Stop after this many matching lines (1-1000).
    """
    base = ctx.workspace.resolve(path, kind="exists")
    try:
        regex = re.compile(
            re.escape(pattern) if literal else pattern, re.IGNORECASE if ignore_case else 0
        )
    except re.error as exc:
        raise ToolError(
            f"invalid regular expression: {exc}. Set literal=true to search for the text as-is."
        ) from exc
    context_lines = clamp(context_lines, 0, 10)
    max_results = clamp(max_results, 1, 1000)
    glob_root = base if base.is_dir() else base.parent
    out: list[str] = []
    matches = files_matched = too_big = 0
    limited = timed_out = False
    deadline = time.monotonic() + _SEARCH_SECONDS

    for scanned, file in enumerate(iter_files(base), 1):
        if scanned > _MAX_FILES_SCANNED or time.monotonic() > deadline:
            timed_out = True
            break
        if glob and not glob_match(file.relative_to(glob_root).as_posix(), glob):
            continue
        if not ctx.workspace.contains(file.resolve()):
            continue  # a symlink pointing outside the workspace
        try:
            if file.stat().st_size > _SEARCH_MAX_FILE:
                too_big += 1
                continue
        except OSError:
            continue
        if is_binary(file):
            continue
        try:
            lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        hits = [i for i, line in enumerate(lines) if regex.search(line)]
        if not hits:
            continue
        files_matched += 1
        shown = ctx.workspace.display(file)
        hit_set = set(hits)
        printed_until = -1
        if context_lines and out:
            out.append("--")
        for i in hits:
            if matches >= max_results:
                limited = True
                break
            low, high = max(0, i - context_lines), min(len(lines) - 1, i + context_lines)
            if context_lines and printed_until >= 0 and low > printed_until + 1:
                out.append("--")
            for j in range(max(low, printed_until + 1), high + 1):
                sep = ":" if j in hit_set else "-"
                out.append(f"{shown}{sep}{j + 1}{sep}{_excerpt(lines[j], regex)}")
            printed_until = max(printed_until, high)
            matches += 1
        if limited:
            break

    notes = []
    if limited:
        notes.append(f"stopped at max_results={max_results}")
    if timed_out:
        notes.append("search stopped early (time or file limit); narrow path or glob")
    if too_big:
        notes.append(f"skipped {too_big} file(s) over {human_size(_SEARCH_MAX_FILE)}")
    suffix = f" ({'; '.join(notes)})" if notes else ""
    where = ctx.workspace.display(base)
    if not matches:
        return f"No matches for {pattern!r} in {where}{suffix}."
    header = f"{matches} matching line(s) in {files_matched} file(s){suffix}:"
    return header + "\n" + "\n".join(out)


def _excerpt(line: str, regex: re.Pattern[str], width: int = 300) -> str:
    if len(line) <= width:
        return line
    match = regex.search(line)
    start = max(0, (match.start() if match else 0) - width // 3)
    piece = line[start : start + width]
    return ("..." if start else "") + piece + ("..." if start + width < len(line) else "")


@tool
def file_info(ctx: ToolContext, path: str) -> dict[str, Any]:
    """Show metadata for a file or directory: type, size, permissions, owner, timestamps,
    MIME type, and the line count of text files.

    Args:
        path: File or directory, relative to the workspace root.
    """
    target = ctx.workspace.resolve(path, kind="exists")
    unresolved = Path(path).expanduser()
    if not unresolved.is_absolute():
        unresolved = ctx.workspace.root / unresolved
    st = target.stat()
    info: dict[str, Any] = {
        "path": ctx.workspace.display(target),
        "type": _file_type(st.st_mode),
        "size": human_size(st.st_size),
        "size_bytes": st.st_size,
        "permissions": stat.filemode(st.st_mode),
        "mode": oct(stat.S_IMODE(st.st_mode)),
        "modified": local_iso(st.st_mtime),
        "accessed": local_iso(st.st_atime),
    }
    owner = _owner(st)
    if owner:
        info["owner"] = owner
    if unresolved.is_symlink():
        info["symlink_target"] = os.readlink(unresolved)
    if stat.S_ISREG(st.st_mode):
        info["mime_type"] = mimetypes.guess_type(target.name)[0] or "unknown"
        info["executable"] = os.access(target, os.X_OK)
        binary = is_binary(target)
        info["binary"] = binary
        if not binary and st.st_size <= _COUNT_LINES_UP_TO:
            with target.open("rb") as fh:
                info["lines"] = sum(1 for _ in fh)
    elif stat.S_ISDIR(st.st_mode):
        info["entries"] = _count_entries(str(target))
    return info


def _file_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        return "device"
    return "other"


def _owner(st: os.stat_result) -> str | None:
    try:
        import grp
        import pwd
    except ImportError:  # Windows
        return None
    try:
        user = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        user = str(st.st_uid)
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = str(st.st_gid)
    return f"{user}:{group}"


@tool
def disk_usage(ctx: ToolContext, path: str = ".", top: int = 15) -> str:
    """Find what's using disk space: the largest entries and files under a directory, plus
    free space on its filesystem. Use it for "why is my disk full?" or "why is this repo so
    big?". Sizes are apparent sizes; other mounted filesystems aren't entered (like du -x).

    Args:
        path: Directory to analyze, relative to the workspace root.
        top: How many of the largest entries and files to list (1-100).
    """
    base = ctx.workspace.resolve(path, kind="dir")
    top = clamp(top, 1, 100)
    device = base.stat().st_dev
    entry_sizes: collections.Counter[tuple[str, bool]] = collections.Counter()
    largest: list[tuple[int, str]] = []
    total = files = 0
    partial = False
    for dirpath, dirnames, filenames in os.walk(base):
        kept = []
        for name in dirnames:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.S_ISDIR(st.st_mode) and st.st_dev == device:
                kept.append(name)
        dirnames[:] = kept
        relative_dir = os.path.relpath(dirpath, base)
        top_dir = None if relative_dir == "." else relative_dir.split(os.sep, 1)[0]
        for name in filenames:
            files += 1
            if files > 2_000_000:
                partial = True
                break
            full = os.path.join(dirpath, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            total += st.st_size
            entry_sizes[(top_dir or name, top_dir is not None)] += st.st_size
            item = (st.st_size, full)
            if len(largest) < top:
                heapq.heappush(largest, item)
            elif item > largest[0]:
                heapq.heapreplace(largest, item)
        if partial:
            break

    shown = ctx.workspace.display(base)
    fs = shutil.disk_usage(base)
    lines = [
        f"Filesystem holding {shown}: {human_size(fs.total)} total, {human_size(fs.used)} used "
        f"({fs.used / fs.total:.0%}), {human_size(fs.free)} free",
        f"Under {shown}: {human_size(total)} in {files:,} files"
        + (" (partial: stopped after 2,000,000 files)" if partial else ""),
        "",
        "Largest entries:",
    ]
    for (name, is_dir), size in entry_sizes.most_common(top):
        lines.append(f"  {human_size(size):>10}  {name}{'/' if is_dir else ''}")
    lines += ["", "Largest files:"]
    for size, full in sorted(largest, reverse=True):
        lines.append(f"  {human_size(size):>10}  {ctx.workspace.display(Path(full))}")
    return "\n".join(lines)


@tool
def diff_files(ctx: ToolContext, path_a: str, path_b: str, context_lines: int = 3) -> str:
    """Compare two text files and show a unified diff (like `diff -u`).

    For changes tracked by git, use git_diff instead.

    Args:
        path_a: The original file.
        path_b: The changed file.
        context_lines: Unchanged lines to show around each change (0-20).
    """
    texts = []
    for path in (path_a, path_b):
        file = ctx.workspace.resolve(path, kind="file")
        if is_binary(file):
            raise ToolError(f"{path} is a binary file; compare checksums with compute_hash instead")
        if file.stat().st_size > _SEARCH_MAX_FILE:
            raise ToolError(f"{path} is too large to diff ({human_size(file.stat().st_size)})")
        texts.append(
            (ctx.workspace.display(file), file.read_text(encoding="utf-8", errors="replace"))
        )
    (name_a, text_a), (name_b, text_b) = texts
    diff = unified_diff(
        text_a, text_b, name_a, name_b, context=clamp(context_lines, 0, 20), max_lines=1500
    )
    return diff or f"{name_a} and {name_b} are identical."


# ------------------------------------------------------------------------ writing


def _read_exact(file: Path) -> str:
    with file.open(encoding="utf-8", errors="strict", newline="") as fh:
        return fh.read()


def _write_exact(file: Path, text: str) -> None:
    with file.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _line_count(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def _write_preview(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    target = ctx.workspace.resolve(args["path"])
    shown = ctx.workspace.display(target)
    content = args["content"]
    if target.is_file():
        old = target.read_text(encoding="utf-8", errors="replace")
        diff = unified_diff(old, content, f"a/{shown}", f"b/{shown}", max_lines=80)
        return diff or f"{shown}: no changes"
    lines = content.splitlines()
    head = "\n".join(f"+{line}" for line in lines[:20])
    more = f"\n... ({len(lines) - 20} more lines)" if len(lines) > 20 else ""
    return f"new file {shown} ({len(lines)} lines)\n{head}{more}"


@tool(risk="write", preview=_write_preview)
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    """Create a file, or replace an existing file's entire content.

    Prefer edit_file for changes to part of an existing file. Missing parent directories
    are created.

    Args:
        path: File to write, relative to the workspace root.
        content: The complete new content of the file.
    """
    target = ctx.workspace.resolve(path)
    if target.is_dir():
        raise ToolError(f"{path} is a directory")
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_exact(target, content)
    verb = "Overwrote" if existed else "Created"
    size = human_size(len(content.encode("utf-8")))
    return f"{verb} {ctx.workspace.display(target)} ({_line_count(content)} lines, {size})."


def _apply_edit(text: str, old: str, new: str, replace_all: bool) -> tuple[str, int]:
    if not old:
        raise ToolError("old_text is empty; use write_file to create or overwrite a whole file")
    count = text.count(old)
    if count == 0 and "\r\n" in text and "\n" in old and "\r\n" not in old:
        # The file uses Windows line endings; match them.
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        count = text.count(old)
    if count == 0:
        raise ToolError(
            "old_text wasn't found in the file. Read the file again and copy the text exactly, "
            "including whitespace and indentation."
        )
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_text appears {count} times; include more surrounding lines so it matches "
            "exactly once, or set replace_all=true"
        )
    if replace_all:
        return text.replace(old, new), count
    return text.replace(old, new, 1), 1


def _load_for_edit(ctx: ToolContext, path: str) -> tuple[Path, str]:
    target = ctx.workspace.resolve(path, kind="file")
    if is_binary(target):
        raise ToolError(f"{path} is a binary file")
    try:
        return target, _read_exact(target)
    except UnicodeDecodeError as exc:
        raise ToolError(f"{path} isn't valid UTF-8 text, so it can't be edited safely") from exc


def _edit_preview(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    target, text = _load_for_edit(ctx, args["path"])
    updated, _ = _apply_edit(
        text, args["old_text"], args["new_text"], bool(args.get("replace_all", False))
    )
    shown = ctx.workspace.display(target)
    return unified_diff(text, updated, f"a/{shown}", f"b/{shown}", max_lines=80)


@tool(risk="write", preview=_edit_preview)
def edit_file(
    ctx: ToolContext, path: str, old_text: str, new_text: str, replace_all: bool = False
) -> str:
    """Replace exact text in a file and show the resulting diff.

    old_text must match the file exactly (whitespace and indentation included) and occur
    exactly once, unless replace_all is true. Read the file first so you can copy it
    precisely; include a few surrounding lines to make the match unique.

    Args:
        path: File to edit, relative to the workspace root.
        old_text: The exact text to replace.
        new_text: The replacement text (may be empty to delete old_text).
        replace_all: Replace every occurrence instead of requiring a unique match.
    """
    target, text = _load_for_edit(ctx, path)
    updated, count = _apply_edit(text, old_text, new_text, replace_all)
    _write_exact(target, updated)
    shown = ctx.workspace.display(target)
    diff = unified_diff(text, updated, f"a/{shown}", f"b/{shown}", max_lines=60)
    return f"Edited {shown} ({count} replacement{'s' if count != 1 else ''}).\n{diff}"


TOOLS = (
    read_file,
    list_dir,
    find_files,
    search_text,
    file_info,
    disk_usage,
    diff_files,
    write_file,
    edit_file,
)
