"""Terminal rendering for the CLI: streamed answers, tool activity, approvals, usage."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from typing import Any, TextIO

from .agent import RunResult
from .events import (
    Event,
    ModelFallback,
    Notice,
    ProgressNote,
    TextDelta,
    ToolCallFinished,
    ToolCallRejected,
    ToolCallStarted,
    ToolUseStarted,
    TurnFinished,
    TurnStarted,
)
from .model import Usage
from .policy import Approval
from .pricing import estimate_cost
from .tools import Tool, ToolCall


def isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def color_enabled(stream: Any, disabled: bool = False) -> bool:
    if disabled or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return isatty(stream) and os.environ.get("TERM") != "dumb"


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _sgr(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled and text else text

    def bold(self, text: str) -> str:
        return self._sgr("1", text)

    def dim(self, text: str) -> str:
        return self._sgr("2", text)

    def italic(self, text: str) -> str:
        return self._sgr("3", text)

    def red(self, text: str) -> str:
        return self._sgr("31", text)

    def green(self, text: str) -> str:
        return self._sgr("32", text)

    def yellow(self, text: str) -> str:
        return self._sgr("33", text)

    def cyan(self, text: str) -> str:
        return self._sgr("36", text)


class Glyphs:
    def __init__(self, unicode: bool) -> None:
        self.call = "●" if unicode else "*"
        self.result = "⎿" if unicode else "->"
        self.note = "·" if unicode else "-"
        self.spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if unicode else "|/-\\"


def supports_unicode(stream: Any) -> bool:
    return "utf" in (getattr(stream, "encoding", "") or "").lower()


class Spinner:
    """An animated status line, shown while waiting on the model or a slow tool."""

    def __init__(self, stream: TextIO, frames: str) -> None:
        self.stream = stream
        self.frames = frames
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._label = ""
        self._since = 0.0

    def start(self, label: str) -> None:
        with self._lock:
            self._label = label
            if self._thread is not None:
                return
            self._since = time.monotonic()
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join()
        with self._lock:
            self._thread = None
            self.stream.write("\r\033[K")
            self.stream.flush()

    def _run(self) -> None:
        frame = 0
        while not self._stop.wait(0.08):
            with self._lock:
                if self._stop.is_set():
                    break
                elapsed = time.monotonic() - self._since
                glyph = self.frames[frame % len(self.frames)]
                self.stream.write(f"\r{glyph} {self._label}… {elapsed:.0f}s\033[K")
                self.stream.flush()
            frame += 1


class Terminal:
    """Asks the user questions on their terminal, even when stdin is a pipe."""

    def __init__(self, stdin: TextIO, prompt_stream: TextIO) -> None:
        self.stdin = stdin
        self.prompt_stream = prompt_stream

    def available(self) -> bool:
        if isatty(self.stdin):
            return True
        try:
            with open("/dev/tty", "r+"):
                return True
        except OSError:
            return False

    def ask(self, prompt: str) -> str | None:
        """Show ``prompt`` and read one line; None if nobody can answer."""
        if isatty(self.stdin):
            self.prompt_stream.write(prompt)
            self.prompt_stream.flush()
            return self.stdin.readline() or None
        try:
            with open("/dev/tty", "r+", encoding="utf-8", errors="replace") as tty:
                tty.write(prompt)
                tty.flush()
                return tty.readline() or None
        except OSError:
            return None


def format_tokens(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.2f}M"


def format_cost(cost: float | None) -> str | None:
    if cost is None:
        return None
    return f"~${cost:.4f}" if cost < 0.01 else f"~${cost:.2f}"


def format_args(args: Mapping[str, Any] | None, width: int = 110) -> str:
    """Compact one-line rendering of tool arguments."""
    if not args:
        return "()"
    parts = []
    for key, value in args.items():
        if isinstance(value, str):
            if "\n" in value:
                text = f"<{value.count(chr(10)) + 1} lines>"
            else:
                text = json.dumps(
                    value if len(value) <= 60 else value[:57] + "...", ensure_ascii=False
                )
        else:
            text = json.dumps(value, ensure_ascii=False, default=str)
            if len(text) > 60:
                text = text[:57] + "..."
        parts.append(f"{key}={text}")
    joined = ", ".join(parts)
    if len(joined) > width:
        joined = joined[: width - 3] + "..."
    return f"({joined})"


def summarize(content: str, limit: int = 100) -> str:
    lines = content.strip().splitlines() or [""]
    first = lines[0].strip()
    if len(first) > limit:
        first = first[: limit - 3] + "..."
    return first + (f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else "")


class Renderer:
    """Turns agent events into terminal output.

    The answer goes to ``out``; tool activity and notices go to ``status``. Interactively
    both are the terminal. When stdout is piped, status goes to stderr and only the final
    answer is written to stdout (``stream_text=False``).
    """

    def __init__(
        self,
        out: TextIO,
        status: TextIO,
        *,
        terminal: Terminal | None = None,
        stream_text: bool = True,
        print_answer: bool = True,
        color: bool = False,
        status_color: bool = False,
        verbose: bool = False,
        spinner: bool = True,
    ) -> None:
        self.out = out
        self.status = status
        self.terminal = terminal
        self.stream_text = stream_text
        self.print_answer = print_answer
        self.style = Style(color)
        self.sstyle = Style(status_color)
        self.verbose = verbose
        self.glyphs = Glyphs(supports_unicode(status))
        self._shared = out is status
        self._out_fresh = True  # out cursor is at the start of a line
        self._status_fresh = True
        self._spinner = Spinner(status, self.glyphs.spinner) if spinner and isatty(status) else None
        self._last_started: str | None = None
        self._run_started = time.monotonic()

    # -- event dispatch ---------------------------------------------------------------

    def __call__(self, event: Event) -> None:
        if isinstance(event, TurnStarted):
            if event.turn == 1:
                self._run_started = time.monotonic()
            self._spin("thinking")
        elif isinstance(event, TextDelta):
            if self.stream_text:
                self._write_out(event.text)
        elif isinstance(event, ProgressNote):
            note = f"{self.glyphs.note} {event.text}"
            self._status_line(self.sstyle.dim(self.sstyle.italic(note)))
            self._spin("working")
        elif isinstance(event, ToolUseStarted):
            self._spin(f"preparing {event.name}")
        elif isinstance(event, ModelFallback):
            self._status_line(
                self.sstyle.yellow(f"! {event.from_model} declined; continuing on {event.to_model}")
            )
        elif isinstance(event, TurnFinished):
            self._spin_stop()
        elif isinstance(event, ToolCallStarted):
            self._on_tool_started(event.call)
        elif isinstance(event, ToolCallFinished):
            self._on_tool_finished(event)
        elif isinstance(event, ToolCallRejected):
            reason = summarize(event.reason, 140)
            self._status_line(
                self.sstyle.yellow(f"  {self.glyphs.result} {event.call.name} not run: {reason}")
            )
        elif isinstance(event, Notice):
            self._status_line(self.sstyle.yellow(f"! {event.message}"))

    def _on_tool_started(self, call: ToolCall) -> None:
        self._last_started = call.id
        s = self.sstyle
        self._status_line(
            f"{s.cyan(self.glyphs.call)} {s.bold(call.name)}{s.dim(format_args(call.args))}"
        )
        self._spin(f"running {call.name}")

    def _on_tool_finished(self, event: ToolCallFinished) -> None:
        result, s = event.result, self.sstyle
        name = "" if event.call.id == self._last_started else f"{event.call.name}: "
        timing = f"  ({result.duration:.1f}s)" if result.duration >= 1 else ""
        line = f"  {self.glyphs.result} {name}{summarize(result.content)}{timing}"
        self._status_line(s.red(line) if result.is_error else s.dim(line))
        if self.verbose and not result.is_error:
            lines = result.content.splitlines()
            for text in lines[1:30]:
                self._status_line(s.dim(f"      {text}"))
            if len(lines) > 30:
                self._status_line(s.dim(f"      ... ({len(lines) - 30} more lines)"))

    # -- approvals --------------------------------------------------------------------

    def approve(self, tool: Tool, call: ToolCall) -> Approval:
        """Ask the user whether a tool call may run (used as the policy's approver)."""
        self._spin_stop()
        s = self.sstyle
        self._status_line(
            f"{s.yellow(s.bold('? Allow ' + tool.name))} {s.dim(f'[{tool.risk.value}]')}"
        )
        body = call.preview or json.dumps(call.args, indent=2, ensure_ascii=False, default=str)
        lines = body.splitlines()
        for text in lines[:80]:
            self._status_line("    " + self._color_diff_line(text))
        if len(lines) > 80:
            self._status_line(s.dim(f"    ... ({len(lines) - 80} more lines)"))
        if self.terminal is None:
            return Approval.DENY
        answer = self.terminal.ask(f"  Run it? [y]es / [n]o / [a]lways allow {tool.name}: ")
        self._status_fresh = True
        choice = (answer or "").strip().lower()
        if choice in ("y", "yes"):
            return Approval.ALLOW
        if choice in ("a", "always"):
            return Approval.ALWAYS
        return Approval.DENY

    def _color_diff_line(self, text: str) -> str:
        s = self.sstyle
        if text.startswith("+++") or text.startswith("---"):
            return s.bold(text)
        if text.startswith("+"):
            return s.green(text)
        if text.startswith("-"):
            return s.red(text)
        if text.startswith("@@"):
            return s.cyan(text)
        return text

    # -- end of a run -----------------------------------------------------------------

    def finish(self, result: RunResult) -> None:
        self._spin_stop()
        if self.stream_text:
            if not self._out_fresh:
                self._write_out("\n")
        elif self.print_answer and result.text.strip():
            self.out.write(result.text.strip() + "\n")
            self.out.flush()
        warning = self._warning(result)
        if warning:
            self._status_line(self.sstyle.yellow(warning))
        elapsed = time.monotonic() - self._run_started
        self._status_line(self.sstyle.dim(self.usage_line(result, elapsed)))

    @staticmethod
    def _warning(result: RunResult) -> str | None:
        if result.stop_reason == "refusal":
            category = result.refusal.category if result.refusal else None
            message = "The model declined this request" + (f" ({category})" if category else "")
            if result.refusal and result.refusal.explanation:
                message += f": {result.refusal.explanation}"
            return message + ". Try rephrasing it, or another model with --model."
        return {
            "max_turns": "Stopped at the turn limit (--max-turns) before finishing.",
            "max_tokens": "The reply was cut off at the output limit (--max-tokens).",
            "interrupted": "Interrupted.",
            "model_context_window_exceeded": (
                "The conversation no longer fits in the context window; start a new one (/clear)."
            ),
        }.get(result.stop_reason)

    @staticmethod
    def usage_line(result: RunResult, elapsed: float | None = None) -> str:
        parts = [f"{result.turns} turn{'s' if result.turns != 1 else ''}"]
        if result.tool_calls:
            count = len(result.tool_calls)
            parts.append(f"{count} tool call{'s' if count != 1 else ''}")
        parts.append(usage_summary(result.usage, result.model))
        if elapsed is not None:
            parts.append(f"{elapsed:.1f}s")
        return " · ".join(parts)

    # -- low-level output -------------------------------------------------------------

    def _spin(self, label: str) -> None:
        if self._spinner is None:
            return
        if not self._status_fresh:
            # The spinner redraws its line with "\r"; never draw over unfinished text.
            self.status.write("\n")
            self._status_fresh = True
            if self._shared:
                self._out_fresh = True
        self._spinner.start(label)

    def _spin_stop(self) -> None:
        if self._spinner is not None:
            self._spinner.stop()

    def _write_out(self, text: str) -> None:
        if not text:
            return
        self._spin_stop()
        self.out.write(text)
        self.out.flush()
        self._out_fresh = text.endswith("\n")
        if self._shared:
            self._status_fresh = self._out_fresh

    def status_line(self, text: str) -> None:
        self._status_line(text)

    def _status_line(self, text: str) -> None:
        self._spin_stop()
        if not self._status_fresh:
            self.status.write("\n")
        self.status.write(text + "\n")
        self.status.flush()
        self._status_fresh = True
        if self._shared:
            self._out_fresh = True


def usage_summary(usage: Usage, model: str) -> str:
    text = f"{format_tokens(usage.prompt_tokens)} in"
    if usage.cache_read_input_tokens:
        text += f" ({format_tokens(usage.cache_read_input_tokens)} cached)"
    text += f" · {format_tokens(usage.output_tokens)} out"
    cost = format_cost(estimate_cost(model, usage))
    return f"{text} · {cost}" if cost else text
