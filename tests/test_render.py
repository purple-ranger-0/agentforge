from __future__ import annotations

import io
from typing import Any

import pytest

from agentforge import Approval, RunResult, ToolCall, ToolResult, Usage, tool
from agentforge import events as ev
from agentforge.model import Refusal
from agentforge.render import Renderer, Spinner, format_args, format_cost, format_tokens, summarize


@tool(risk="write")
def deploy(env: str) -> str:
    """Deploy."""
    return env


class FakeTerminal:
    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def available(self) -> bool:
        return True

    def ask(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        return self.answer


def make(answer: str | None = "y") -> tuple[Renderer, io.StringIO, io.StringIO, FakeTerminal]:
    out, status = io.StringIO(), io.StringIO()
    terminal = FakeTerminal(answer)
    renderer = Renderer(out, status, terminal=terminal, stream_text=False)  # type: ignore[arg-type]
    return renderer, out, status, terminal


def result(**kwargs: Any) -> RunResult:
    defaults: dict[str, Any] = {
        "text": "Answer.",
        "stop_reason": "end_turn",
        "turns": 2,
        "tool_calls": [],
        "usage": Usage(1200, 340, 50_000, 0),
        "model": "claude-opus-5-5",
    }
    return RunResult(**{**defaults, **kwargs})


def test_format_helpers() -> None:
    assert (
        format_tokens(999) == "999"
        and format_tokens(51_200) == "51.2k"
        and format_tokens(3_400_000) == "3.40M"
    )
    assert (
        format_cost(None) is None
        and format_cost(0.004) == "~$0.0040"
        and format_cost(1.5) == "~$1.50"
    )
    assert format_args({"path": "a.py", "n": 3}) == '(path="a.py", n=3)'
    assert format_args({"content": "a\nb\nc"}) == "(content=<3 lines>)"
    assert format_args({"q": "x" * 100}) == f'(q="{"x" * 57}...")'
    assert format_args({}) == "()"
    assert summarize("first\nsecond\nthird") == "first  (+2 lines)"
    assert summarize("y" * 200, 10) == "yyyyyyy..."


def test_tool_activity_goes_to_the_status_stream() -> None:
    renderer, out, status, _ = make()
    call = ToolCall("t1", "read_file", {"path": "a.py"}, args={"path": "a.py"})
    renderer(ev.TurnStarted(1))
    renderer(ev.TextDelta("streamed text is not echoed in pipe mode"))
    renderer(ev.ProgressNote("Looking at a.py"))
    renderer(ev.ToolCallStarted(call, deploy))
    renderer(ev.ToolCallFinished(call, ToolResult("a.py (lines 1-3 of 3)\n1\n2", duration=1.25)))
    renderer(ev.ToolCallRejected(ToolCall("t2", "x", {}), "Permission denied: no."))
    renderer(ev.ModelFallback("claude-opus-5-5", "claude-opus-5"))
    renderer(ev.Notice("retrying"))
    renderer.finish(result())
    assert out.getvalue() == "Answer.\n"
    assert status.getvalue().splitlines() == [
        "- Looking at a.py",
        '* read_file(path="a.py")',
        "  -> a.py (lines 1-3 of 3)  (+2 lines)  (1.2s)",
        "  -> x not run: Permission denied: no.",
        "! claude-opus-5-5 declined; continuing on claude-opus-5",
        "! retrying",
        status.getvalue().splitlines()[-1],
    ]
    assert (
        status.getvalue()
        .splitlines()[-1]
        .startswith("2 turns · 51.2k in (50.0k cached) · 340 out · ~$0.02 · ")
    )


def test_interactive_text_and_status_share_lines_cleanly() -> None:
    term = io.StringIO()
    renderer = Renderer(term, term)
    renderer(ev.TextDelta("Let me check"))
    call = ToolCall("t1", "git_status", {}, args={})
    renderer(ev.ToolCallStarted(call, deploy))
    renderer(ev.ToolCallFinished(call, ToolResult("clean", is_error=True)))
    renderer(ev.TextDelta("All good."))
    renderer.finish(result(text="Let me checkAll good."))
    lines = term.getvalue().splitlines()
    assert lines[:4] == ["Let me check", "* git_status()", "  -> clean", "All good."]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("y\n", Approval.ALLOW),
        ("always\n", Approval.ALWAYS),
        ("n\n", Approval.DENY),
        ("", Approval.DENY),
        (None, Approval.DENY),
    ],
)
def test_approvals(answer: str | None, expected: Approval) -> None:
    renderer, _, status, terminal = make(answer)
    call = ToolCall(
        "t1", "deploy", {"env": "prod"}, args={"env": "prod"}, preview="--- a\n+++ b\n-old\n+new"
    )
    assert renderer.approve(deploy, call) is expected
    assert status.getvalue().splitlines() == [
        "? Allow deploy [write]",
        "    --- a",
        "    +++ b",
        "    -old",
        "    +new",
    ]
    assert terminal.prompts == ["  Run it? [y]es / [n]o / [a]lways allow deploy: "]


def test_approval_without_preview_shows_arguments() -> None:
    renderer, _, status, _ = make("y")
    renderer.approve(deploy, ToolCall("t1", "deploy", {"env": "prod"}, args={"env": "prod"}))
    assert '    "env": "prod"' in status.getvalue()


@pytest.mark.parametrize(
    ("kwargs", "warning"),
    [
        (
            {"stop_reason": "refusal", "refusal": Refusal("bio", "Not allowed.")},
            "The model declined this request (bio): Not allowed.",
        ),
        ({"stop_reason": "max_turns"}, "Stopped at the turn limit"),
        ({"stop_reason": "max_tokens"}, "cut off at the output limit"),
        ({"stop_reason": "interrupted"}, "Interrupted."),
    ],
)
def test_finish_warnings(kwargs: dict[str, Any], warning: str) -> None:
    renderer, _, status, _ = make()
    renderer.finish(result(**kwargs))
    assert warning in status.getvalue()


def test_spinner_clears_its_line() -> None:
    stream = io.StringIO()
    spinner = Spinner(stream, "ab")
    spinner.start("thinking")
    spinner.start("still thinking")  # relabels, doesn't start a second thread
    spinner.stop()
    spinner.stop()
    assert stream.getvalue().endswith("\r\033[K")
