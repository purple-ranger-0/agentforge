from __future__ import annotations

import asyncio
import dataclasses
from typing import Annotated, Literal

import pytest

from agentforge import Risk, Tool, ToolContext, ToolResult, tool
from agentforge.errors import ToolDefinitionError, ToolError, ToolInputError
from agentforge.tools import to_result, truncate


@tool
def greet(name: str, excited: bool = False) -> str:
    """Greet someone.

    Args:
        name: Who to greet.
        excited: Add an exclamation mark.
    """
    return f"Hello, {name}{'!' if excited else '.'}"


@tool(name="where_am_i", risk="read")
def workspace_root(ctx: ToolContext) -> str:
    """Report the workspace root."""
    return str(ctx.workspace.root)


@tool(risk="write")
async def slow_double(n: Annotated[int, "The number to double."]) -> int:
    """Double a number, asynchronously."""
    await asyncio.sleep(0)
    return n * 2


def test_decorator_builds_definition() -> None:
    assert isinstance(greet, Tool)
    assert greet.to_param() == {
        "name": "greet",
        "description": "Greet someone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Who to greet."},
                "excited": {
                    "type": "boolean",
                    "description": "Add an exclamation mark.",
                    "default": False,
                },
            },
            "required": ["name"],
        },
    }
    assert greet.risk is Risk.READ
    assert greet.parallel_safe


def test_decorated_function_stays_callable() -> None:
    assert greet("Ada") == "Hello, Ada."
    assert greet.__name__ == "greet"
    assert "Greet someone" in (greet.__doc__ or "")


def test_context_parameter_is_injected_not_exposed(ctx: ToolContext) -> None:
    assert workspace_root.name == "where_am_i"
    assert workspace_root.input_schema == {"type": "object", "properties": {}}
    assert workspace_root.invoke({}, ctx) == str(ctx.workspace.root)


def test_async_tools_run(ctx: ToolContext) -> None:
    assert slow_double.is_async
    assert slow_double.risk is Risk.WRITE
    assert not slow_double.parallel_safe
    assert slow_double.input_schema["properties"]["n"] == {
        "type": "integer",
        "description": "The number to double.",
    }
    assert slow_double.invoke(slow_double.validate({"n": "21"}), ctx) == 42


def test_async_tool_works_inside_a_running_event_loop(ctx: ToolContext) -> None:
    async def caller() -> int:
        return slow_double.invoke({"n": 4}, ctx)

    assert asyncio.run(caller()) == 8


def test_validate_rejects_unknown_and_missing_params() -> None:
    with pytest.raises(
        ToolInputError, match=r"unknown parameter\(s\): mood \(valid parameters: excited, name\)"
    ):
        greet.validate({"name": "x", "mood": "happy"})
    with pytest.raises(ToolInputError, match="missing required parameter 'name'"):
        greet.validate({})
    with pytest.raises(ToolInputError, match="must be a JSON object"):
        greet.validate(["name"])
    assert workspace_root.validate(None) == {}  # None input means "no arguments"


def test_tool_needs_a_description() -> None:
    def undocumented(x: int) -> int:
        return x

    with pytest.raises(ToolDefinitionError, match="needs a description"):
        tool(undocumented)
    assert tool(description="Identity.")(undocumented).description == "Identity."


def noop() -> None:
    """Do nothing."""


@pytest.mark.parametrize("bad_name", ["has space", "x" * 65, "dots.not.allowed"])
def test_tool_names_are_validated(bad_name: str) -> None:
    with pytest.raises(ToolDefinitionError, match="invalid tool name"):
        tool(name=bad_name)(noop)


def test_var_args_are_rejected() -> None:
    def splat(*items: str) -> None:
        """Doc."""

    with pytest.raises(ToolDefinitionError, match="regular named parameter"):
        tool(splat)


def test_risk_and_parallel_safety_overrides() -> None:
    @tool(risk="network", parallel_safe=False)
    def ping(host: str) -> str:
        """Ping."""
        return host

    assert ping.risk is Risk.NETWORK
    assert not ping.parallel_safe
    with pytest.raises(ValueError):
        tool(risk="dangerous")(noop)


def test_preview_errors() -> None:
    def broken(ctx: ToolContext, args: dict) -> str:
        raise RuntimeError("boom")

    def refusing(ctx: ToolContext, args: dict) -> str:
        raise ToolError("can't do that")

    @tool(risk="write", preview=broken)
    def a() -> None:
        """A."""

    @tool(risk="write", preview=refusing)
    def b() -> None:
        """B."""

    assert a.render_preview(None, {}) == "(preview unavailable: boom)"  # type: ignore[arg-type]
    with pytest.raises(ToolError):
        b.render_preview(None, {})  # type: ignore[arg-type]
    assert greet.render_preview(None, {}) is None  # type: ignore[arg-type]


@dataclasses.dataclass
class Point:
    x: int
    y: int


@pytest.mark.parametrize(
    ("value", "content"),
    [
        ("text", "text"),
        ("   ", "(empty output)"),
        (None, "(no output)"),
        (b"bytes", "bytes"),
        ({"b": 1, "a": [1, 2]}, '{\n  "b": 1,\n  "a": [\n    1,\n    2\n  ]\n}'),
        (42, "42"),
        (Point(1, 2), '{\n  "x": 1,\n  "y": 2\n}'),
    ],
)
def test_to_result(value: object, content: str) -> None:
    assert to_result(value).content == content


def test_tool_result_blocks() -> None:
    ok = ToolResult("fine")
    assert ok.to_block("id1") == {"type": "tool_result", "tool_use_id": "id1", "content": "fine"}
    assert to_result(ToolResult("bad", is_error=True)).to_block("id2")["is_error"] is True


def test_truncate_keeps_head_and_tail() -> None:
    text = "H" * 600 + "M" * 1000 + "T" * 400
    out = truncate(text, 1000)
    assert out.startswith("H" * 600)
    assert out.endswith("T" * 300)
    assert "1,000 characters omitted" in out
    assert truncate("short", 1000) == "short"


def test_literal_parameters_document_choices() -> None:
    @tool
    def pick(color: Literal["red", "blue"] = "red") -> str:
        """Pick a color."""
        return color

    assert pick.input_schema["properties"]["color"] == {
        "type": "string",
        "enum": ["red", "blue"],
        "default": "red",
    }
    with pytest.raises(ToolInputError, match='expected one of "red", "blue"'):
        pick.validate({"color": "green"})
