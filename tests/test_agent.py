from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agentforge import (
    Agent,
    AnthropicModel,
    Approval,
    MalformedToolInput,
    ModelRequest,
    PermissionPolicy,
    ToolContext,
    ToolError,
    tool,
)
from agentforge import events as ev
from agentforge.agent import echo_safe
from agentforge.testing import ScriptedModel, call, refusal, reply, say, thinking

barrier = threading.Barrier(2)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def meet(name: str) -> str:
    """Wait until a second call arrives (proves calls run concurrently)."""
    barrier.wait(timeout=5)
    return f"met {name}"


order: list[str] = []


@tool(risk="write")
def record(label: str) -> str:
    """Record a label (a side effect, so calls must run one at a time)."""
    order.append(f"start {label}")
    time.sleep(0.02)
    order.append(f"end {label}")
    return label


@tool
def fail(kind: str) -> str:
    """Fail on purpose."""
    if kind == "tool":
        raise ToolError("the widget is offline")
    raise ValueError("unexpected")


@tool
def remember(ctx: ToolContext, key: str, value: str | None = None) -> str:
    """Store or read a value in session state."""
    if value is not None:
        ctx.state[key] = value
        return "stored"
    return ctx.state.get(key, "(unset)")


def make_agent(
    script: list[Any], tools: list[Any] | None = None, **kwargs: Any
) -> tuple[Agent, ScriptedModel, list[Any]]:
    model = ScriptedModel(script)
    seen: list[Any] = []
    agent = Agent(
        tools if tools is not None else [add],
        model=model,
        on_event=seen.append,
        environment=kwargs.pop("environment", False),
        **kwargs,
    )
    return agent, model, seen


def test_plain_answer() -> None:
    agent, model, seen = make_agent(["Hi there."])
    result = agent.run("hello")
    assert result.text == "Hi there."
    assert result.stop_reason == "end_turn"
    assert result.completed
    assert result.turns == 1
    assert str(result) == "Hi there."
    assert [type(e) for e in seen] == [ev.TurnStarted, ev.TextDelta, ev.TurnFinished]
    assert model.requests[0].messages == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_tool_round_trip() -> None:
    agent, model, _ = make_agent([reply("Adding.", call("add", a=2, b="3")), say("5")])
    result = agent.run("2+3?")
    assert result.text == "5"
    assert [(c.name, c.args, c.result.content) for c in result.tool_calls] == [
        ("add", {"a": 2, "b": 3}, "5")
    ]
    second = model.requests[1].messages
    assert second[1]["role"] == "assistant"
    assert second[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_scripted_001", "content": "5"}],
    }


def test_results_for_parallel_calls_share_one_message_in_order() -> None:
    agent, model, _ = make_agent([reply(call("add", a=1, b=1), call("add", a=2, b=2)), say("ok")])
    agent.run("go")
    results = model.requests[1].messages[-1]["content"]
    assert [r["content"] for r in results] == ["2", "4"]
    assert [r["tool_use_id"] for r in results] == ["toolu_scripted_001", "toolu_scripted_002"]


def test_read_only_calls_run_concurrently() -> None:
    barrier.reset()
    agent, _, _ = make_agent(
        [reply(call("meet", name="a"), call("meet", name="b")), say("ok")], tools=[meet]
    )
    result = agent.run("go")
    assert [c.result.content for c in result.tool_calls] == ["met a", "met b"]


def test_side_effect_calls_run_one_at_a_time_in_order() -> None:
    order.clear()
    script = [reply(call("record", label="1"), call("record", label="2")), say("ok")]
    agent, _, _ = make_agent(script, tools=[record], policy=PermissionPolicy("auto"))
    agent.run("go")
    assert order == ["start 1", "end 1", "start 2", "end 2"]


def test_mixed_batches_keep_their_order() -> None:
    order.clear()
    script = [
        reply(
            call("add", a=1, b=0),
            call("add", a=2, b=0),
            call("record", label="w"),
            call("add", a=3, b=0),
        ),
        say("ok"),
    ]
    agent, _, seen = make_agent(script, tools=[add, record], policy=PermissionPolicy("auto"))
    agent.run("go")
    lifecycle = [
        (type(e).__name__, e.call.args.get("a", e.call.args.get("label")))
        for e in seen
        if isinstance(e, (ev.ToolCallStarted, ev.ToolCallFinished))
    ]
    assert lifecycle == [
        ("ToolCallStarted", 1),
        ("ToolCallStarted", 2),
        ("ToolCallFinished", 1),
        ("ToolCallFinished", 2),
        ("ToolCallStarted", "w"),
        ("ToolCallFinished", "w"),
        ("ToolCallStarted", 3),
        ("ToolCallFinished", 3),
    ]


def test_unknown_tool_and_invalid_input_become_error_results() -> None:
    agent, model, seen = make_agent(
        [reply(call("nope"), call("add", a="x", b=1), call("add", a=1)), say("sorry")]
    )
    result = agent.run("go")
    contents = [c.result.content for c in result.tool_calls]
    assert contents[0] == "Unknown tool 'nope'. Available tools: add."
    assert contents[1].startswith("Invalid input for add: a: expected integer, got string")
    assert "missing required parameter 'b'" in contents[2]
    assert all(c.result.is_error for c in result.tool_calls)
    assert sum(isinstance(e, ev.ToolCallRejected) for e in seen) == 3
    assert all(r["is_error"] for r in model.requests[1].messages[-1]["content"])


def test_tool_exceptions_become_error_results() -> None:
    agent, _, _ = make_agent(
        [reply(call("fail", kind="tool"), call("fail", kind="bug")), say("ok")], tools=[fail]
    )
    result = agent.run("go")
    assert [(c.result.content, c.result.is_error) for c in result.tool_calls] == [
        ("the widget is offline", True),
        ("ValueError: unexpected", True),
    ]


def test_denied_calls_are_reported_to_the_model() -> None:
    previews = []

    def show(ctx: ToolContext, args: dict) -> str:
        return f"would record {args['label']}"

    @tool(risk="write", preview=show)
    def guarded(label: str) -> str:
        """Guarded."""
        raise AssertionError("must not run")

    def approver(t, c):
        previews.append(c.preview)
        return Approval.DENY

    agent, _, _ = make_agent(
        [call("guarded", label="x"), say("ok")],
        tools=[guarded],
        policy=PermissionPolicy(approver=approver),
    )
    result = agent.run("go")
    assert previews == ["would record x"]
    assert result.tool_calls[0].result.content.startswith("Permission denied: the user declined")


def test_a_preview_that_fails_rejects_without_asking() -> None:
    def impossible(ctx: ToolContext, args: dict) -> str:
        raise ToolError("old_text wasn't found")

    @tool(risk="write", preview=impossible)
    def edit() -> str:
        """Edit."""
        return "edited"

    def approver(t, c):
        raise AssertionError("shouldn't ask")

    agent, _, _ = make_agent(
        [call("edit"), say("ok")], tools=[edit], policy=PermissionPolicy(approver=approver)
    )
    assert agent.run("go").tool_calls[0].result.content == "old_text wasn't found"


def test_max_turns() -> None:
    agent, _, _ = make_agent([call("add", a=1, b=1)] * 3, max_turns=2)
    result = agent.run("loop")
    assert result.stop_reason == "max_turns"
    assert result.turns == 2
    assert not result.completed


def test_refusal_is_not_added_to_history() -> None:
    agent, model, _ = make_agent([refusal("cyber", "nope"), say("Sure.")])
    session = agent.session()
    result = session.send("first")
    assert result.stop_reason == "refusal"
    assert result.refusal.category == "cyber"
    assert result.text == ""
    assert [m["role"] for m in session.messages] == ["user"]
    session.send("second")
    assert [m["role"] for m in model.requests[1].messages] == ["user", "user"]


def test_tool_calls_cut_off_by_max_tokens_are_not_run() -> None:
    agent, _, _ = make_agent(
        [reply(call("add", a=1, b=2), stop_reason="max_tokens"), say("retrying")]
    )
    result = agent.run("go")
    assert result.tool_calls[0].result.is_error
    assert "max_tokens" in result.tool_calls[0].result.content
    assert result.text == "retrying"


def test_tool_calls_cut_off_by_the_context_window_are_not_run() -> None:
    script = [reply(call("add", a=1, b=2), stop_reason="model_context_window_exceeded"), say("ok")]
    agent, _, _ = make_agent(script)
    assert agent.run("go").tool_calls[0].result.is_error


def test_pause_turn_is_resumed_by_resending() -> None:
    agent, model, _ = make_agent(
        [reply("Searching...", stop_reason="pause_turn"), say("Found it.")]
    )
    result = agent.run("go")
    assert result.text == "Found it."
    assert [m["role"] for m in model.requests[1].messages] == ["user", "assistant"]


def test_malformed_tool_input_is_retried() -> None:
    def broken(request: ModelRequest) -> Any:
        raise MalformedToolInput("bad json")

    agent, _, seen = make_agent([broken, say("ok")])
    assert agent.run("go").text == "ok"
    assert any(isinstance(e, ev.Notice) for e in seen)
    agent, _, _ = make_agent([broken, broken, broken])
    with pytest.raises(MalformedToolInput):
        agent.run("go")


def test_interrupting_a_tool_keeps_history_valid() -> None:
    @tool
    def hang() -> str:
        """Hang."""
        raise KeyboardInterrupt

    agent, _, _ = make_agent(
        [reply(call("hang"), call("add", a=1, b=1))], tools=[hang, add], parallel_tools=False
    )
    session = agent.session()
    result = session.send("go")
    assert result.stop_reason == "interrupted"
    results = session.messages[-1]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert all("Interrupted" in r["content"] for r in results)


def test_interrupting_the_model_appends_nothing() -> None:
    def ctrl_c(request: ModelRequest) -> Any:
        raise KeyboardInterrupt

    agent, _, _ = make_agent([ctrl_c])
    session = agent.session()
    assert session.send("go").stop_reason == "interrupted"
    assert [m["role"] for m in session.messages] == ["user"]


def test_large_results_are_truncated() -> None:
    @tool
    def big() -> str:
        """Big."""
        return "x" * 5000

    agent, model, _ = make_agent([call("big"), say("ok")], tools=[big], max_result_chars=1000)
    agent.run("go")
    content = model.requests[1].messages[-1]["content"][0]["content"]
    assert len(content) < 1200
    assert "characters omitted" in content


def test_session_tools_are_sorted_filtered_and_stable() -> None:
    script = [call("add", a=1, b=1), say("ok")]
    agent, model, _ = make_agent(
        script, tools=[record, add, meet], policy=PermissionPolicy("readonly")
    )
    agent.run("go")
    names = [[t["name"] for t in r.tools] for r in model.requests]
    assert names == [["add", "meet"], ["add", "meet"]]


def test_environment_note_only_in_the_first_message(tmp_path: Path) -> None:
    agent, model, _ = make_agent(["one", "two"], environment=True, workspace=tmp_path)
    session = agent.session()
    session.send("a")
    session.send("b")
    first, _, second = model.requests[1].messages
    assert first["content"][0]["text"].startswith("<environment>\nWorkspace root: " + str(tmp_path))
    assert len(second["content"]) == 1


def test_tool_state_is_shared_within_a_session() -> None:
    script = [call("remember", key="k", value="v"), call("remember", key="k"), say("done")]
    agent, _, _ = make_agent(script, tools=[remember])
    result = agent.run("go")
    assert [c.result.content for c in result.tool_calls] == ["stored", "v"]


def test_transcript_round_trips_as_json(tmp_path: Path) -> None:
    agent, _, _ = make_agent([call("add", a=1, b=2), say("3")])
    session = agent.session()
    session.send("1+2")
    path = session.save(tmp_path / "t.json")
    data = json.loads(path.read_text())
    assert data["model"] == "scripted"
    assert data["tools"][0]["name"] == "add"
    assert [m["role"] for m in data["messages"]] == ["user", "assistant", "user", "assistant"]


def test_content_block_prompts_and_empty_prompts() -> None:
    agent, model, _ = make_agent(["ok"])
    agent.run([{"type": "text", "text": "custom"}])
    assert model.requests[0].messages[0]["content"] == [{"type": "text", "text": "custom"}]
    with pytest.raises(ValueError):
        agent.run("   ")


def test_agent_validates_tools_and_accepts_model_names() -> None:
    with pytest.raises(TypeError, match="decorate the function with @tool"):
        Agent([print])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="duplicate tool name"):
        Agent([add, add], model=ScriptedModel([]))
    agent = Agent([add], model="claude-sonnet-5-5")
    assert isinstance(agent.model, AnthropicModel)
    assert agent.model.model == "claude-sonnet-5-5"


def test_progress_notes_are_emitted() -> None:
    agent, _, seen = make_agent(
        [reply(thinking("Checking the math."), call("add", a=1, b=1)), say("2")]
    )
    agent.run("go")
    assert ev.ProgressNote("Checking the math.") in seen


# ------------------------------------------------------------------ fallback boundary


def test_echo_safe_without_fallback_is_identity() -> None:
    content = [
        {"type": "text", "text": "a"},
        {"type": "tool_use", "id": "1", "name": "x", "input": {}},
    ]
    assert echo_safe(content) == content


def test_echo_safe_drops_the_declined_attempt() -> None:
    content = [
        {"type": "thinking", "thinking": "", "signature": "s1"},
        {"type": "text", "text": "Partial answer "},
        {"type": "tool_use", "id": "old", "name": "add", "input": {"a": 1}},
        {"type": "server_tool_use", "id": "srv1", "name": "web_search", "input": {}},
        {"type": "web_search_tool_result", "tool_use_id": "srv1", "content": []},
        {"type": "server_tool_use", "id": "srv2", "name": "web_search", "input": {}},
        {
            "type": "fallback",
            "from": {"model": "claude-opus-5-5"},
            "to": {"model": "claude-opus-5"},
        },
        {"type": "text", "text": "continued."},
        {"type": "tool_use", "id": "new", "name": "add", "input": {"a": 1, "b": 2}},
    ]
    assert [b.get("id") or b.get("tool_use_id") or b["type"] for b in echo_safe(content)] == [
        "text",
        "srv1",
        "srv1",
        "text",
        "new",
    ]


def test_tool_calls_before_a_fallback_never_run() -> None:
    turn = reply(
        call("add", a=100, b=100),
        {"type": "fallback", "from": {"model": "m1"}, "to": {"model": "m2"}},
        call("add", a=1, b=2),
    )
    agent, model, seen = make_agent([turn, say("3")])
    result = agent.run("go")
    assert [c.args for c in result.tool_calls] == [{"a": 1, "b": 2}]
    assert ev.ModelFallback("m1", "m2") in seen
    assistant = model.requests[1].messages[1]["content"]
    assert [b["type"] for b in assistant] == ["tool_use"]
