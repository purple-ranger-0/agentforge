"""Events emitted while an agent works; pass ``on_event=`` to ``Agent`` to receive them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Usage
    from .tools import Tool, ToolCall, ToolResult


@dataclass(frozen=True)
class TurnStarted:
    """A model request is about to be sent."""

    turn: int


@dataclass(frozen=True)
class TextDelta:
    """A chunk of the model's reply text, as it streams."""

    text: str


@dataclass(frozen=True)
class ProgressNote:
    """A short note the model wrote between tool calls ("Checking the logs next")."""

    text: str


@dataclass(frozen=True)
class ToolUseStarted:
    """The model started writing a call to ``name``; its input is still streaming."""

    name: str


@dataclass(frozen=True)
class ModelFallback:
    """The requested model declined and the API continued the turn on another model."""

    from_model: str
    to_model: str


@dataclass(frozen=True)
class TurnFinished:
    """A model response is complete."""

    turn: int
    stop_reason: str | None
    usage: Usage
    model: str


@dataclass(frozen=True)
class ToolCallStarted:
    call: ToolCall
    tool: Tool


@dataclass(frozen=True)
class ToolCallFinished:
    call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class ToolCallRejected:
    """A tool call didn't run: unknown tool, invalid input, or permission denied."""

    call: ToolCall
    reason: str


@dataclass(frozen=True)
class Notice:
    """Something the user should know about (a retry, for example)."""

    message: str


Event = (
    TurnStarted
    | TextDelta
    | ProgressNote
    | ToolUseStarted
    | ModelFallback
    | TurnFinished
    | ToolCallStarted
    | ToolCallFinished
    | ToolCallRejected
    | Notice
)
