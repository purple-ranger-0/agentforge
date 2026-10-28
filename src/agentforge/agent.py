"""The agent loop: send the conversation to the model, run the tools it asks for, repeat."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .errors import MalformedToolInput, ToolError, ToolInputError
from .events import (
    Event,
    Notice,
    ToolCallFinished,
    ToolCallRejected,
    ToolCallStarted,
    TurnFinished,
    TurnStarted,
)
from .model import DEFAULT_MODEL, AnthropicModel, ModelBackend, ModelRequest, Refusal, Usage
from .policy import PermissionPolicy
from .prompts import SYSTEM_PROMPT, environment_note
from .tools import Tool, ToolCall, ToolContext, ToolResult, to_result, truncate
from .workspace import Workspace

logger = logging.getLogger("agentforge")

_MAX_MALFORMED_RETRIES = 2
_MAX_PARALLEL_TOOLS = 8


@dataclass
class RunResult:
    """The outcome of :meth:`Session.send` / :meth:`Agent.run`.

    ``stop_reason`` is the model's final stop reason (``"end_turn"`` normally), or one of
    ``"max_turns"``, ``"refusal"`` or ``"interrupted"``.
    """

    text: str
    stop_reason: str
    turns: int
    tool_calls: list[ToolCall]
    usage: Usage
    model: str
    refusal: Refusal | None = None

    @property
    def completed(self) -> bool:
        return self.stop_reason in ("end_turn", "stop_sequence")

    def __str__(self) -> str:
        return self.text


class Agent:
    """A model, a set of tools, and the rules for using them.

    >>> agent = Agent(tools=[word_count])
    >>> print(agent.run("How many words are in 'to be or not to be'?").text)
    """

    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        model: ModelBackend | str | None = None,
        system: str | None = None,
        policy: PermissionPolicy | None = None,
        workspace: Workspace | str | os.PathLike[str] | None = None,
        max_turns: int = 50,
        on_event: Callable[[Event], None] | None = None,
        parallel_tools: bool = True,
        max_result_chars: int = 40_000,
        environment: bool = True,
    ) -> None:
        self.tools: dict[str, Tool] = {}
        for t in tools:
            if not isinstance(t, Tool):
                raise TypeError(f"{t!r} is not a Tool; decorate the function with @tool")
            if t.name in self.tools:
                raise ValueError(f"duplicate tool name {t.name!r}")
            self.tools[t.name] = t
        if model is None or isinstance(model, str):
            model = AnthropicModel(model or DEFAULT_MODEL)
        self.model = model
        self.system = SYSTEM_PROMPT if system is None else system
        self.policy = policy or PermissionPolicy()
        if not isinstance(workspace, Workspace):
            workspace = Workspace(workspace or ".")
        self.workspace = workspace
        self.max_turns = max_turns
        self.on_event = on_event
        self.parallel_tools = parallel_tools
        self.max_result_chars = max_result_chars
        self.environment = environment

    def run(self, prompt: str | list[dict[str, Any]]) -> RunResult:
        """Answer one prompt in a fresh conversation."""
        return self.session().send(prompt)

    def session(self) -> Session:
        """Start a multi-turn conversation."""
        return Session(self)

    def emit(self, event: Event) -> None:
        if self.on_event is not None:
            self.on_event(event)


class Session:
    """A multi-turn conversation with an :class:`Agent`. The history is append-only."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        exposed = sorted(
            (t for t in agent.tools.values() if agent.policy.exposes(t)), key=lambda t: t.name
        )
        # Fixed for the whole session: tool definitions sit at the very start of the prompt,
        # so changing them mid-conversation would throw away the prompt cache.
        self.tools = {t.name: t for t in exposed}
        self.tool_params = [t.to_param() for t in exposed]
        self.messages: list[dict[str, Any]] = []
        self.usage = Usage()
        self.state: dict[str, Any] = {}

    def send(self, prompt: str | list[dict[str, Any]]) -> RunResult:
        """Add a user message and let the agent work until it answers."""
        agent = self.agent
        self.messages.append({"role": "user", "content": self._user_content(prompt)})
        calls: list[ToolCall] = []
        usage = Usage()
        model_name = agent.model.model
        turns = 0
        malformed = 0
        last_text = ""

        def finish(stop_reason: str, refusal: Refusal | None = None) -> RunResult:
            return RunResult(last_text, stop_reason, turns, calls, usage, model_name, refusal)

        while True:
            if turns >= agent.max_turns:
                return finish("max_turns")
            request = ModelRequest(
                system=agent.system, tools=self.tool_params, messages=list(self.messages)
            )
            agent.emit(TurnStarted(turns + 1))
            try:
                response = agent.model.generate(request, agent.emit)
            except MalformedToolInput:
                malformed += 1
                if malformed > _MAX_MALFORMED_RETRIES:
                    raise
                agent.emit(Notice("The model sent malformed tool input; retrying that turn."))
                continue
            except KeyboardInterrupt:
                return finish("interrupted")
            turns += 1
            usage += response.usage
            self.usage += response.usage
            model_name = response.model or model_name
            agent.emit(TurnFinished(turns, response.stop_reason, response.usage, model_name))

            if response.stop_reason == "refusal":
                # Don't keep a declined turn: its partial output isn't an answer, and a
                # cut-off tool call without a result would break the next request.
                last_text = ""
                return finish("refusal", response.refusal)

            content = echo_safe(response.content)
            if content:
                self.messages.append({"role": "assistant", "content": content})
            last_text = _text(content)
            if response.stop_reason == "pause_turn":
                continue  # a server-side tool paused mid-turn; sending the turn back resumes it
            tool_uses = [block for block in content if block.get("type") == "tool_use"]
            if not tool_uses:
                return finish(response.stop_reason or "end_turn")
            cut_off = response.stop_reason in ("max_tokens", "model_context_window_exceeded")
            results, interrupted = self._run_tools(tool_uses, truncated=cut_off, calls=calls)
            self.messages.append({"role": "user", "content": results})
            if interrupted:
                return finish("interrupted")

    # -- tools -------------------------------------------------------------------------

    def _run_tools(
        self, tool_uses: list[dict[str, Any]], *, truncated: bool, calls: list[ToolCall]
    ) -> tuple[list[dict[str, Any]], bool]:
        round_calls = [
            ToolCall(id=block["id"], name=block.get("name", ""), input=block.get("input") or {})
            for block in tool_uses
        ]
        calls.extend(round_calls)
        interrupted = False
        try:
            approved = []
            for call in round_calls:
                tool = self._prepare(call, truncated)
                if tool is not None:
                    approved.append((call, tool))
            self._execute(approved)
        except KeyboardInterrupt:
            interrupted = True
        # Every tool_use needs a tool_result, including calls that never ran.
        for call in round_calls:
            if call.result is None:
                call.result = ToolResult(
                    "Interrupted: the user stopped this before it finished.", is_error=True
                )
        return [call.result.to_block(call.id) for call in round_calls], interrupted

    def _prepare(self, call: ToolCall, truncated: bool) -> Tool | None:
        """Validate and authorize a call. On failure, record an error result and return None."""
        tool = self.tools.get(call.name)
        if tool is None:
            available = ", ".join(self.tools) or "none"
            return self._reject(call, f"Unknown tool {call.name!r}. Available tools: {available}.")
        if truncated:
            return self._reject(
                call,
                "This call was cut off because the response hit the max_tokens limit, so it "
                "wasn't run. Send less input per call (for example, write a large file in "
                "several smaller edits).",
            )
        try:
            call.args = tool.validate(call.input)
        except ToolInputError as exc:
            return self._reject(
                call, f"Invalid input for {tool.name}: {exc}. Fix the arguments and try again."
            )
        policy = self.agent.policy
        if policy.requires_approval(tool):
            try:
                call.preview = tool.render_preview(self._context(call), call.args)
            except ToolError as exc:
                return self._reject(call, str(exc))
        allowed, reason = policy.check(tool, call)
        if not allowed:
            return self._reject(
                call,
                f"Permission denied: {reason}. Don't retry this exact call; adjust your "
                "approach or ask the user.",
            )
        return tool

    def _reject(self, call: ToolCall, message: str) -> None:
        call.result = ToolResult(message, is_error=True)
        self.agent.emit(ToolCallRejected(call, message))

    def _execute(self, approved: list[tuple[ToolCall, Tool]]) -> None:
        """Run calls in order; consecutive parallel-safe calls run concurrently."""
        batch: list[tuple[ToolCall, Tool]] = []
        for call, tool in approved:
            if tool.parallel_safe and self.agent.parallel_tools:
                batch.append((call, tool))
                continue
            self._run_batch(batch)
            batch = []
            self._run_one(call, tool)
        self._run_batch(batch)

    def _run_batch(self, batch: list[tuple[ToolCall, Tool]]) -> None:
        if len(batch) <= 1:
            for call, tool in batch:
                self._run_one(call, tool)
            return
        emit = self.agent.emit
        for call, tool in batch:
            emit(ToolCallStarted(call, tool))
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(_MAX_PARALLEL_TOOLS, len(batch)), thread_name_prefix="agentforge-tool"
        )
        try:
            futures = [pool.submit(self._invoke, call, tool) for call, tool in batch]
            for (call, _), future in zip(batch, futures, strict=True):
                call.result = future.result()
                emit(ToolCallFinished(call, call.result))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _run_one(self, call: ToolCall, tool: Tool) -> None:
        self.agent.emit(ToolCallStarted(call, tool))
        call.result = self._invoke(call, tool)
        self.agent.emit(ToolCallFinished(call, call.result))

    def _invoke(self, call: ToolCall, tool: Tool) -> ToolResult:
        started = time.monotonic()
        try:
            result = to_result(tool.invoke(call.args or {}, self._context(call)))
        except ToolError as exc:
            result = ToolResult(str(exc) or type(exc).__name__, is_error=True)
        except Exception as exc:
            logger.debug("tool %s raised", call.name, exc_info=True)
            result = ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
        result.content = truncate(result.content, self.agent.max_result_chars) or "(empty output)"
        result.duration = time.monotonic() - started
        return result

    def _context(self, call: ToolCall) -> ToolContext:
        return ToolContext(workspace=self.agent.workspace, call_id=call.id, state=self.state)

    # -- messages ----------------------------------------------------------------------

    def _user_content(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if self.agent.environment and not self.messages:
            blocks.append({"type": "text", "text": environment_note(self.agent.workspace)})
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("prompt is empty")
            blocks.append({"type": "text", "text": prompt})
        else:
            blocks.extend(prompt)
        return blocks

    def transcript(self) -> dict[str, Any]:
        """Everything needed to inspect (or replay) this conversation."""
        return {
            "agentforge": __version__,
            "model": self.agent.model.model,
            "system": self.agent.system,
            "tools": self.tool_params,
            "messages": self.messages,
            "usage": self.usage.to_dict(),
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the transcript as JSON."""
        target = Path(path)
        data = json.dumps(self.transcript(), indent=2, ensure_ascii=False, default=str)
        target.write_text(data + "\n", encoding="utf-8")
        return target


def echo_safe(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare an assistant turn for the history after a mid-response model fallback.

    When the requested model declines partway through, the API continues on a fallback
    model and marks the switch with a ``fallback`` block. Before the last such block only
    text (and completed server-tool pairs) may be sent back: thinking and tool calls from
    the declined attempt must be dropped, and those tool calls must not run.
    """
    last = max((i for i, b in enumerate(content) if b.get("type") == "fallback"), default=None)
    if last is None:
        return list(content)
    before, after = content[:last], content[last + 1 :]
    server_calls = {b.get("id") for b in before if b.get("type") == "server_tool_use"}
    answered = {
        b.get("tool_use_id")
        for b in before
        if str(b.get("type", "")).endswith("_tool_result") and b.get("tool_use_id") in server_calls
    }
    kept = []
    for block in before:
        kind = block.get("type")
        if (
            kind == "text"
            or (kind == "server_tool_use" and block.get("id") in answered)
            or (str(kind).endswith("_tool_result") and block.get("tool_use_id") in answered)
        ):
            kept.append(block)
    return kept + [b for b in after if b.get("type") != "fallback"]


def _text(content: list[dict[str, Any]]) -> str:
    return "".join(block.get("text", "") for block in content if block.get("type") == "text")
