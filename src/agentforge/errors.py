"""Exceptions raised by AgentForge."""

from __future__ import annotations


class AgentForgeError(Exception):
    """Base class for AgentForge errors."""


class ToolError(AgentForgeError):
    """Raise inside a tool to report a failure to the model.

    The message goes back to the model as an error tool result (no traceback), so write
    it for the model: say what went wrong and, when you can, what to try instead.
    """


class ToolDefinitionError(AgentForgeError, TypeError):
    """A function can't be turned into a tool (no docstring, unsupported annotation, ...)."""


class ToolInputError(AgentForgeError, ValueError):
    """Model-supplied tool input doesn't match the tool's parameters."""


class ModelError(AgentForgeError):
    """A model backend failed in a way the agent loop knows how to handle."""


class MalformedToolInput(ModelError):
    """The model streamed tool-call JSON that couldn't be parsed. Re-sending the request is safe."""


class MissingCredentials(ModelError):
    """No Anthropic API credentials could be found."""
