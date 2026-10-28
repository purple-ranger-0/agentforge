"""Run shell commands in the workspace (needs approval unless the user allows ``exec``)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from typing import IO, Any

from ..errors import ToolError
from ..tools import ToolContext, tool
from ._util import clamp, strip_ansi

_KEEP_BYTES = 100_000  # output kept from both the start and the end of a long run


class _OutputCollector(threading.Thread):
    """Drains a pipe, keeping only the head and tail of very long output."""

    def __init__(self, stream: IO[bytes]) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def run(self) -> None:
        fd = self.stream.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            self.total += len(chunk)
            room = _KEEP_BYTES - len(self.head)
            if room > 0:
                self.head += chunk[:room]
                chunk = chunk[room:]
            if chunk:
                self.tail += chunk
                if len(self.tail) > _KEEP_BYTES:
                    del self.tail[: len(self.tail) - _KEEP_BYTES]

    def text(self) -> str:
        omitted = self.total - len(self.head) - len(self.tail)
        middle = (
            f"\n[... {omitted:,} bytes of output omitted ...]\n".encode() if omitted > 0 else b""
        )
        return (bytes(self.head) + middle + bytes(self.tail)).decode("utf-8", errors="replace")


def _kill(proc: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)  # the whole process group, children included
    except (ProcessLookupError, PermissionError):
        pass


def _command_preview(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    workdir = ctx.workspace.resolve(args.get("cwd", "."), kind="dir")
    return (
        f"$ {args['command']}\n"
        f"(in {ctx.workspace.display(workdir)}, timeout {args.get('timeout', 120)}s)"
    )


@tool(risk="exec", preview=_command_preview)
def run_command(ctx: ToolContext, command: str, timeout: int = 120, cwd: str = ".") -> str:
    """Run a shell command and return its exit code and combined stdout/stderr.

    Use it for what the other tools can't do: running tests, builds, linters, package
    managers, docker, kubectl, and so on. Prefer the dedicated read-only tools for reading
    files, searching and git history. Commands run non-interactively with stdin closed, so
    pass flags like --yes where a prompt would appear, and avoid editors and pagers.

    Args:
        command: The command line to run (bash syntax).
        timeout: Seconds before the command is killed (1-1800).
        cwd: Working directory, relative to the workspace root.
    """
    if not command.strip():
        raise ToolError("command is empty")
    workdir = ctx.workspace.resolve(cwd, kind="dir")
    timeout = clamp(timeout, 1, 1800)
    env = {
        **os.environ,
        "PAGER": "cat",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    if os.name == "nt":
        popen: dict[str, Any] = {"args": command, "shell": True}
    else:
        shell = shutil.which("bash") or "/bin/sh"
        popen = {"args": [shell, "-c", command], "start_new_session": True}
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            **popen,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        raise ToolError(f"couldn't start the command: {exc}") from exc
    assert proc.stdout is not None
    collector = _OutputCollector(proc.stdout)
    collector.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(proc)
        proc.wait()
    except KeyboardInterrupt:
        _kill(proc)
        proc.wait()
        raise
    finally:
        # A background process may keep the pipe open; don't wait for it forever. Only
        # close our end once the reader is done, so its file descriptor can't be reused
        # under it.
        collector.join(timeout=5)
        if not collector.is_alive():
            proc.stdout.close()
    elapsed = time.monotonic() - started
    output = strip_ansi(collector.text()).rstrip()
    status = f"killed after the {timeout}s timeout" if timed_out else f"exit code {proc.returncode}"
    return f"[{status}, {elapsed:.1f}s]\n{output or '(no output)'}"


TOOLS = (run_command,)
