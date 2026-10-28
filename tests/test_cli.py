from __future__ import annotations

import builtins
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from agentforge import MissingCredentials, ModelRequest, cli
from agentforge.testing import ScriptedModel, call, refusal, reply, say

CUSTOM_TOOLS = '''
from agentforge import tool

@tool
def shout(text: str) -> str:
    """Upper-case some text."""
    return text.upper()
'''


class FakeTTY(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


def run_cli(argv: list[str], model: Any = None, stdin: str = "") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, model=model, stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize(
    ("argv", "routed"),
    [
        ([], ["chat"]),
        (["why is the sky blue"], ["ask", "why is the sky blue"]),
        (["-m", "claude-sonnet-5-5"], ["chat", "-m", "claude-sonnet-5-5"]),
        (["tools", "--json"], ["tools", "--json"]),
        (["--help"], ["--help"]),
    ],
)
def test_route(argv: list[str], routed: list[str]) -> None:
    assert cli.route(argv) == routed


def test_version() -> None:
    code, out, _ = run_cli(["version"])
    assert code == 0 and out.startswith("agentforge 0.1.0")


def test_tools_listing_formats() -> None:
    code, out, _ = run_cli(["tools"])
    assert code == 0
    assert out.startswith("fs (9)\n  read_file ")
    assert "\nRisk levels:" in out
    code, out, _ = run_cli(["tools", "--json", "-t", "data,git_log"])
    tools = json.loads(out)
    assert [t["name"] for t in tools][-1] == "git_log"
    assert {t["pack"] for t in tools} == {"data", "git"}
    assert tools[0]["input_schema"]["type"] == "object"
    code, out, _ = run_cli(["tools", "--markdown", "-t", "shell"])
    assert out.splitlines() == [
        "| Tool | Pack | Risk | What it does |",
        "|---|---|---|---|",
        "| `run_command` | shell | exec | Run a shell command and return its exit code and "
        "combined stdout/stderr. |",
    ]


def test_unknown_pack_is_a_usage_error() -> None:
    code, _, err = run_cli(["tools", "-t", "fs,nope"])
    assert code == 2
    assert "unknown tool pack or tool 'nope'" in err


def test_load_custom_tools(tmp_path: Path) -> None:
    path = tmp_path / "my_tools.py"
    path.write_text(CUSTOM_TOOLS)
    code, out, _ = run_cli(["tools", "-t", "data", "--load", str(path)])
    assert "custom (1)\n  shout " in out
    model = ScriptedModel([call("shout", text="hi"), say("HI")])
    code, out, _ = run_cli(["ask", "-t", "data", "--load", str(path), "shout hi"], model=model)
    assert code == 0 and out == "HI\n"
    assert "shout" in [t["name"] for t in model.requests[0].tools]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "no such file"),
        ("x = 1\n", "no @tool functions found"),
        ("raise RuntimeError('boom')\n", "RuntimeError: boom"),
    ],
)
def test_load_errors(tmp_path: Path, content: str | None, message: str) -> None:
    path = tmp_path / "bad.py"
    if content is not None:
        path.write_text(content)
    code, _, err = run_cli(["tools", "--load", str(path)])
    assert code == 2 and message in err


def test_ask_in_a_pipe_prints_only_the_answer(tmp_path: Path) -> None:
    model = ScriptedModel(
        [reply("Checking.", call("calculate", expression="6*7")), say("It's 42.")]
    )
    code, out, err = run_cli(["ask", "-C", str(tmp_path), "what", "is", "6*7?"], model=model)
    assert code == 0
    assert out == "It's 42.\n"
    assert 'calculate(expression="6*7")' in err
    assert "2 turns · 1 tool call" in err
    prompt = model.requests[0].messages[0]["content"]
    assert prompt[0]["text"].startswith("<environment>")
    assert prompt[1]["text"] == "what is 6*7?"


def test_ask_attaches_stdin_before_the_question() -> None:
    model = ScriptedModel(["Looks fine."])
    run_cli(["ask", "review this"], model=model, stdin="diff --git a/x b/x\n")
    assert model.requests[0].messages[0]["content"][-1]["text"] == (
        "<stdin>\ndiff --git a/x b/x\n</stdin>\n\nreview this"
    )


def test_ask_with_only_stdin_explains_it() -> None:
    model = ScriptedModel(["It's a stack trace."])
    run_cli(["ask"], model=model, stdin="Traceback (most recent call last):\n")
    text = model.requests[0].messages[0]["content"][-1]["text"]
    assert text.endswith("Explain what it is and point out anything that needs attention.")


def test_huge_stdin_is_trimmed_with_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "MAX_STDIN_CHARS", 100)
    model = ScriptedModel(["ok"])
    _, _, err = run_cli(["ask", "summarize"], model=model, stdin="A" * 500 + "Z" * 500)
    assert "stdin has 1,000 characters" in err
    text = model.requests[0].messages[0]["content"][-1]["text"]
    assert "[... 900 characters omitted ...]" in text and text.count("A") == 50


def test_nothing_to_ask() -> None:
    code, _, err = run_cli(["ask"])
    assert code == 2 and "nothing to ask" in err


def test_json_output() -> None:
    model = ScriptedModel([call("calculate", expression="1+1"), say("2")])
    _, out, err = run_cli(["ask", "--json", "1+1"], model=model)
    result = json.loads(out)
    assert result["text"] == "2" and result["stop_reason"] == "end_turn"
    assert result["tool_calls"] == [
        {
            "name": "calculate",
            "input": {"expression": "1+1"},
            "is_error": False,
            "result": "1+1 = 2",
        }
    ]
    assert "1 tool call" in err


def test_risky_tools_without_a_terminal_are_refused(tmp_path: Path) -> None:
    model = ScriptedModel(
        [call("write_file", path="x.txt", content="hi"), say("I couldn't write it.")]
    )
    code, _, err = run_cli(["ask", "-C", str(tmp_path), "write x.txt"], model=model)
    assert code == 0
    assert not (tmp_path / "x.txt").exists()
    result = model.requests[1].messages[-1]["content"][0]
    assert result["is_error"] and "--yes or --allow write_file" in result["content"]
    assert "write_file not run: Permission denied" in err


def test_yes_and_allow_let_risky_tools_run(tmp_path: Path) -> None:
    model = ScriptedModel([call("write_file", path="x.txt", content="hi"), say("done")])
    run_cli(["ask", "-y", "-C", str(tmp_path), "write"], model=model)
    assert (tmp_path / "x.txt").read_text() == "hi"
    model = ScriptedModel([call("write_file", path="y.txt", content="yo"), say("done")])
    run_cli(["ask", "--allow", "write", "-C", str(tmp_path), "write"], model=model)
    assert (tmp_path / "y.txt").read_text() == "yo"


def test_readonly_and_deny_limit_the_tools_offered() -> None:
    model = ScriptedModel(["ok"])
    run_cli(["ask", "--mode", "readonly", "hi"], model=model)
    risks = {t["name"] for t in model.requests[0].tools}
    assert "read_file" in risks and "write_file" not in risks and "run_command" not in risks
    model = ScriptedModel(["ok"])
    run_cli(["ask", "--deny", "exec,network", "hi"], model=model)
    names = {t["name"] for t in model.requests[0].tools}
    assert "run_command" not in names and "fetch_url" not in names and "write_file" in names


def test_save_transcript(tmp_path: Path) -> None:
    target = tmp_path / "t.json"
    run_cli(["ask", "--save", str(target), "hi"], model=ScriptedModel(["hello"]))
    assert json.loads(target.read_text())["messages"][-1]["content"] == [
        {"type": "text", "text": "hello"}
    ]


def test_exit_codes_for_incomplete_runs() -> None:
    code, out, err = run_cli(["ask", "hack it"], model=ScriptedModel([refusal("cyber")]))
    assert code == 3 and out == ""
    assert "The model declined this request (cyber)" in err
    code, _, err = run_cli(
        ["ask", "--max-turns", "1", "loop"],
        model=ScriptedModel([call("calculate", expression="1")]),
    )
    assert code == 3 and "turn limit" in err


def raising(exc: BaseException) -> Any:
    def step(request: ModelRequest) -> Any:
        raise exc

    return step


def api_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (
            MissingCredentials("No Anthropic API credentials found."),
            "error: No Anthropic API credentials found.",
        ),
        (api_error(anthropic.RateLimitError, 429), "error: rate limited"),
        (api_error(anthropic.AuthenticationError, 401), "error: authentication failed"),
        (api_error(anthropic.BadRequestError, 400), "try --compat"),
        (api_error(anthropic.NotFoundError, 404), "a valid model ID"),
        (api_error(anthropic.InternalServerError, 500), "error: API error 500"),
    ],
)
def test_api_errors_are_reported_cleanly(exc: BaseException, message: str) -> None:
    code, out, err = run_cli(["ask", "hi"], model=ScriptedModel([raising(exc)]))
    assert code == 1 and out == ""
    assert message in err


def test_bad_root_and_chat_without_a_terminal(tmp_path: Path) -> None:
    code, _, err = run_cli(["ask", "-C", str(tmp_path / "missing"), "hi"], model=ScriptedModel([]))
    assert code == 2 and "not a directory" in err
    code, _, err = run_cli(["chat"], model=ScriptedModel([]))
    assert code == 2 and "needs an interactive terminal" in err


@pytest.fixture
def scripted_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[list[str]]:
    lines: list[str] = []
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("NO_COLOR", "1")

    def fake_input(prompt: str = "") -> str:
        if not lines:
            raise EOFError
        return lines.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)
    yield lines


def test_chat_session(scripted_input: list[str], tmp_path: Path) -> None:
    scripted_input.extend(
        [
            "what's 2 + \\",
            "2?",
            "/usage",
            "/tools",
            "/bogus",
            f"/save {tmp_path / 's.json'}",
            "/clear",
            "again",
            "/exit",
        ]
    )
    model = ScriptedModel(
        [reply(call("calculate", expression="2+2")), say("4"), say("Hello again.")]
    )
    out = FakeTTY()
    code = cli.main(
        ["chat", "-t", "data", "-C", str(tmp_path)],
        model=model,
        stdin=FakeTTY(),
        stdout=out,
        stderr=io.StringIO(),
    )
    text = out.getvalue()
    assert code == 0
    assert "AgentForge 0.1.0 · scripted · 10 tools · approvals: ask" in text
    assert model.requests[0].messages[0]["content"][-1]["text"] == "what's 2 + \n2?"
    assert "4\n" in text
    assert "calculate         read" in text
    assert "Unknown command /bogus" in text
    assert "Saved the conversation to" in text and (tmp_path / "s.json").exists()
    assert "Started a new conversation." in text
    assert "Hello again." in text
    assert [m["role"] for m in model.requests[-1].messages] == ["user"]  # /clear started fresh


def test_describe_error_falls_back() -> None:
    assert cli.describe_error(RuntimeError("x"), "m") is None
