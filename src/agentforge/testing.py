"""Deterministic, offline model backends for testing agents and tools.

    from agentforge import Agent, tool
    from agentforge.testing import ScriptedModel, call, say

    def test_adds():
        model = ScriptedModel([call("add", a=2, b=3), say("2 + 3 = 5")])
        result = Agent([add], model=model).run("What's 2 + 3?")
        assert result.text == "2 + 3 = 5"
        assert result.tool_calls[0].result.content == "5"

Each scripted step is one model turn: a string (a text reply), a ``call(...)`` block, a
``reply(...)`` with several blocks, or a function that receives the :class:`ModelRequest`
and returns any of those, for assertions and dynamic replies.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import Any

from .events import Event, ModelFallback, ProgressNote, TextDelta, ToolUseStarted
from .model import ModelRequest, ModelResponse, Refusal, Usage

Step = str | dict[str, Any] | ModelResponse | Callable[[ModelRequest], Any]


def say(text: str) -> ModelResponse:
    """A turn where the model replies with text and stops."""
    return reply(text)


def call(name: str, /, **input: Any) -> dict[str, Any]:
    """A ``tool_use`` block. Use it as a step on its own, or inside :func:`reply`."""
    return {"type": "tool_use", "name": name, "input": input}


def thinking(text: str = "") -> dict[str, Any]:
    """A thinking block. Non-empty text plays the part of a progress note."""
    return {"type": "thinking", "thinking": text, "signature": "scripted"}


def reply(
    *parts: str | dict[str, Any], stop_reason: str | None = None, usage: Usage | None = None
) -> ModelResponse:
    """A turn made of several blocks, e.g. ``reply("Checking both files.", call(...), call(...))``.

    The stop reason defaults to ``"tool_use"`` when the turn contains a tool call, otherwise
    ``"end_turn"``.
    """
    content = [{"type": "text", "text": p} if isinstance(p, str) else dict(p) for p in parts]
    if stop_reason is None:
        has_call = any(b.get("type") == "tool_use" for b in content)
        stop_reason = "tool_use" if has_call else "end_turn"
    return ModelResponse(content, stop_reason, "scripted", usage or Usage())


def refusal(category: str | None = None, explanation: str | None = None) -> ModelResponse:
    """A turn where a safety classifier declines the request."""
    return ModelResponse([], "refusal", "scripted", Usage(), Refusal(category, explanation))


class ScriptedModel:
    """Plays back a fixed script of model turns and records every request it receives."""

    def __init__(self, script: Iterable[Step], *, model: str = "scripted") -> None:
        self.model = model
        self._script = list(script)
        self.requests: list[ModelRequest] = []
        self._ids = 0

    @property
    def remaining(self) -> int:
        """How many scripted turns haven't been used yet."""
        return len(self._script)

    def generate(self, request: ModelRequest, emit: Callable[[Event], None]) -> ModelResponse:
        self.requests.append(copy.deepcopy(request))
        if not self._script:
            raise AssertionError(
                f"ScriptedModel ran out of turns: request #{len(self.requests)} has no reply"
            )
        step = self._script.pop(0)
        if callable(step):
            step = step(request)
        response = self._normalize(step)
        for block in response.content:
            kind = block.get("type")
            if kind == "text":
                emit(TextDelta(block["text"]))
            elif kind == "tool_use":
                emit(ToolUseStarted(block["name"]))
            elif kind == "thinking" and block.get("thinking"):
                emit(ProgressNote(block["thinking"]))
            elif kind == "fallback":
                emit(ModelFallback(block["from"]["model"], block["to"]["model"]))
        return response

    def _normalize(self, step: Any) -> ModelResponse:
        if isinstance(step, str):
            response = say(step)
        elif isinstance(step, dict):
            response = reply(step)
        elif isinstance(step, ModelResponse):
            response = copy.deepcopy(step)
        else:
            raise TypeError(f"unsupported script step: {step!r}")
        for block in response.content:
            if block.get("type") == "tool_use" and "id" not in block:
                self._ids += 1
                block["id"] = f"toolu_scripted_{self._ids:03d}"
        if response.model == "scripted":
            response.model = self.model
        return response
