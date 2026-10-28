"""Tools: plain Python functions the model can call.

    @tool
    def word_count(text: str) -> int:
        \"\"\"Count the words in a piece of text.\"\"\"
        return len(text.split())

The function's signature becomes the tool's JSON Schema and its docstring becomes the
description (``Args:`` entries describe the parameters). Decorated functions stay callable.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import functools
import inspect
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import ToolDefinitionError, ToolError, ToolInputError
from .policy import Risk
from .schema import Param, coerce, input_schema, parse_docstring, resolve_hints
from .workspace import Workspace

_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass
class ToolContext:
    """Runtime context for tools that declare a parameter annotated ``ToolContext``.

    ``state`` is shared by all tool calls in one session, for tools that need memory.
    """

    workspace: Workspace
    call_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """What a tool call produced. Return one from a tool to control ``is_error`` yourself."""

    content: str
    is_error: bool = False
    duration: float = 0.0

    def to_block(self, tool_use_id: str) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            block["is_error"] = True
        return block


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    input: dict[str, Any]
    args: dict[str, Any] | None = None  # validated arguments; None if validation failed
    result: ToolResult | None = None
    preview: str | None = None  # shown to the user when asking for approval


Preview = Callable[[ToolContext, Mapping[str, Any]], str]


class Tool:
    """A function plus everything the model needs to call it."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        risk: Risk | str = Risk.READ,
        parallel_safe: bool | None = None,
        preview: Preview | None = None,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        if not _NAME.match(self.name):
            raise ToolDefinitionError(
                f"invalid tool name {self.name!r}: use 1-64 letters, digits, '_' or '-'"
            )
        doc_description, param_docs = parse_docstring(func.__doc__)
        self.description = (description or doc_description).strip()
        if not self.description:
            raise ToolDefinitionError(
                f"tool {self.name!r} needs a description: add a docstring or pass description="
            )
        self.risk = Risk(risk)
        # Read-only and network tools have no local side effects, so they can run concurrently.
        if parallel_safe is None:
            parallel_safe = self.risk in (Risk.READ, Risk.NETWORK)
        self.parallel_safe = parallel_safe
        self.preview = preview
        self.is_async = inspect.iscoroutinefunction(func)

        hints = resolve_hints(func)
        self.context_param: str | None = None
        self.params: list[Param] = []
        for p in inspect.signature(func).parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD, p.POSITIONAL_ONLY):
                raise ToolDefinitionError(
                    f"tool {self.name!r}: parameter {p.name!r} must be a regular named parameter"
                )
            annotation = hints.get(p.name, Any)
            if annotation is ToolContext:
                self.context_param = p.name
                continue
            self.params.append(Param(p.name, annotation, p.default, param_docs.get(p.name)))
        self.input_schema = input_schema(self.params)
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Tool {self.name} ({self.risk.value})>"

    def to_param(self) -> dict[str, Any]:
        """The tool definition sent to the model."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def validate(self, raw: Any) -> dict[str, Any]:
        """Check model-supplied input and convert it into keyword arguments."""
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ToolInputError("tool input must be a JSON object")
        known = {p.name for p in self.params}
        unknown = sorted(set(raw) - known)
        if unknown:
            valid = ", ".join(sorted(known)) or "none"
            raise ToolInputError(
                f"unknown parameter(s): {', '.join(unknown)} (valid parameters: {valid})"
            )
        args: dict[str, Any] = {}
        for param in self.params:
            if param.name in raw:
                args[param.name] = coerce(raw[param.name], param.annotation, param.name)
            elif param.required:
                raise ToolInputError(f"missing required parameter '{param.name}'")
        return args

    def invoke(self, args: Mapping[str, Any], context: ToolContext) -> Any:
        """Run the tool with validated arguments."""
        kwargs = dict(args)
        if self.context_param:
            kwargs[self.context_param] = context
        if self.is_async:
            # A fresh thread has no running event loop, so this also works when the caller
            # is inside one (e.g. a Jupyter notebook).
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.func(**kwargs)).result()
        return self.func(**kwargs)

    def render_preview(self, context: ToolContext, args: Mapping[str, Any]) -> str | None:
        """Describe what a call would do, for the approval prompt.

        A preview may raise :class:`ToolError` when the call can't work (an edit whose text
        isn't in the file); the call is then rejected without asking the user.
        """
        if self.preview is None:
            return None
        try:
            return self.preview(context, args)
        except ToolError:
            raise
        except Exception as exc:  # a broken preview shouldn't block the approval prompt
            return f"(preview unavailable: {exc})"


def tool(
    func: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    risk: Risk | str = Risk.READ,
    parallel_safe: bool | None = None,
    preview: Preview | None = None,
) -> Any:
    """Turn a function into a :class:`Tool`. Use as ``@tool`` or ``@tool(risk="write", ...)``.

    ``risk`` is one of ``"read"`` (default), ``"write"``, ``"network"`` or ``"exec"``; any
    risk other than ``"read"`` asks the user before running (see ``PermissionPolicy``).
    ``preview(context, args) -> str`` renders what the call will do for that prompt.
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(
            f,
            name=name,
            description=description,
            risk=risk,
            parallel_safe=parallel_safe,
            preview=preview,
        )

    return wrap(func) if func is not None else wrap


def to_result(value: Any) -> ToolResult:
    """Convert whatever a tool returned into a ToolResult."""
    if isinstance(value, ToolResult):
        return value
    if value is None:
        return ToolResult("(no output)")
    if isinstance(value, str):
        return ToolResult(value if value.strip() else "(empty output)")
    if isinstance(value, (bytes, bytearray)):
        return ToolResult(bytes(value).decode("utf-8", errors="replace") or "(empty output)")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    return ToolResult(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def truncate(text: str, limit: int) -> str:
    """Keep the head and tail of an over-long tool result."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n\n[... {omitted:,} characters omitted; narrow the request to see "
        f"this part ...]\n\n{text[-tail:]}"
    )
