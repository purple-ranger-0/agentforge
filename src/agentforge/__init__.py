"""AgentForge: tool-calling AI agents in plain Python.

from agentforge import Agent, tool

@tool
def word_count(text: str) -> int:
    \"\"\"Count the words in a piece of text.\"\"\"
    return len(text.split())

print(Agent([word_count]).run("How many words in 'to be or not to be'?").text)
"""

from ._version import __version__
from .agent import Agent, RunResult, Session
from .errors import (
    AgentForgeError,
    MalformedToolInput,
    MissingCredentials,
    ModelError,
    ToolDefinitionError,
    ToolError,
    ToolInputError,
)
from .model import (
    DEFAULT_MODEL,
    AnthropicModel,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    Refusal,
    Usage,
)
from .policy import Approval, PermissionPolicy, Risk
from .tools import Tool, ToolCall, ToolContext, ToolResult, tool
from .workspace import Workspace

__all__ = [
    "DEFAULT_MODEL",
    "Agent",
    "AgentForgeError",
    "AnthropicModel",
    "Approval",
    "MalformedToolInput",
    "MissingCredentials",
    "ModelBackend",
    "ModelError",
    "ModelRequest",
    "ModelResponse",
    "PermissionPolicy",
    "Refusal",
    "Risk",
    "RunResult",
    "Session",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolDefinitionError",
    "ToolError",
    "ToolInputError",
    "ToolResult",
    "Usage",
    "Workspace",
    "__version__",
    "tool",
]
