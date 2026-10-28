"""Model backends: the part of the agent that talks to an LLM.

A backend turns a :class:`ModelRequest` (system prompt, tool definitions, conversation)
into a :class:`ModelResponse`, emitting streaming events along the way. Content blocks are
plain dicts in the Messages API wire format, so conversations can be saved as JSON and
replayed by :class:`agentforge.testing.ScriptedModel`.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .errors import MalformedToolInput, MissingCredentials
from .events import ModelFallback, ProgressNote, TextDelta, ToolUseStarted

if TYPE_CHECKING:
    from .events import Event

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 64_000  # room for adaptive thinking plus a long reply; requests stream
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

_BETA_FALLBACKS = "server-side-fallback-2026-07-01"
_BETA_PROGRESS_UPDATES = "thinking-display-updates-2026-08-18"

# Models that return progress notes between tool calls with thinking.display = "updates".
_PROGRESS_UPDATE_MODELS = frozenset(
    {
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-mythos-5-1",
    }
)
# Models that accept `fallbacks: "default"` (server-side retry when a safety classifier declines).
_FALLBACK_MODELS = frozenset(
    {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5"}
)
# Older models that predate adaptive thinking and the effort parameter.
_LEGACY_MODEL = re.compile(
    r"^claude-(?:[0-3]|instant|(?:opus|sonnet|haiku)-4(?:-[015])?(?:-\d{8})?$)"
)

_CREDENTIALS_HELP = (
    "No Anthropic API credentials found. Set ANTHROPIC_API_KEY (get a key at "
    "https://platform.claude.com) or sign in with `ant auth login`."
)


@dataclass
class Usage:
    """Token counts for one or more model requests."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )

    @property
    def prompt_tokens(self) -> int:
        """Total prompt size, cached and uncached."""
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class ModelRequest:
    system: str
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]


@dataclass
class Refusal:
    """Why the model (or a safety classifier) declined a request."""

    category: str | None = None
    explanation: str | None = None


@dataclass
class ModelResponse:
    content: list[dict[str, Any]]
    stop_reason: str | None
    model: str
    usage: Usage = field(default_factory=Usage)
    refusal: Refusal | None = None


class ModelBackend(Protocol):
    """Anything that can produce the next assistant turn."""

    model: str

    def generate(self, request: ModelRequest, emit: Callable[[Event], None]) -> ModelResponse: ...


@dataclass(frozen=True)
class Capabilities:
    adaptive_thinking: bool
    effort: bool
    progress_updates: bool
    fallbacks: bool


def capabilities(model: str) -> Capabilities:
    """What request features a model ID supports (unknown future models get the modern basics)."""
    name = model.removeprefix("anthropic.")
    legacy = bool(_LEGACY_MODEL.match(name))
    return Capabilities(
        adaptive_thinking=not legacy,
        effort=not legacy,
        progress_updates=name in _PROGRESS_UPDATE_MODELS,
        fallbacks=name in _FALLBACK_MODELS,
    )


class AnthropicModel:
    """Claude via the official ``anthropic`` SDK, streamed.

    Every request uses adaptive thinking with an explicit ``effort`` and prompt caching
    (a breakpoint on the system prompt plus automatic caching of the conversation tail).
    Against the first-party API it also opts into three request features, which ``betas``
    switches off for gateways or proxies that reject them:

    * ``fallbacks: "default"``: if a safety classifier declines a request, the API retries
      it on Anthropic's recommended fallback model instead of failing the turn.
    * ``thinking.display: "updates"``: the short notes Claude writes between tool calls
      come back as text, so long agentic turns don't look frozen.
    * ``eager_input_streaming`` on tools: large tool inputs (file contents) stream as they
      are generated. The agent validates every tool input before running anything.

    ``betas=None`` (the default) enables them unless ``ANTHROPIC_BASE_URL`` or the client's
    base URL points somewhere other than ``api.anthropic.com``.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str | None = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any = None,
        betas: bool | None = None,
        max_retries: int = 4,
    ) -> None:
        if effort is not None and effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}, not {effort!r}")
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.capabilities = capabilities(model)
        self._client = client
        self._betas = betas
        self._max_retries = max_retries

    def __repr__(self) -> str:
        return f"AnthropicModel({self.model!r}, effort={self.effort!r})"

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(max_retries=self._max_retries)
        return self._client

    @property
    def betas(self) -> bool:
        if self._betas is not None:
            return self._betas
        if self._client is None:
            base_url = os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        else:
            base_url = str(getattr(self._client, "base_url", ""))
        return "api.anthropic.com" in base_url

    def build_params(self, request: ModelRequest) -> dict[str, Any]:
        """The keyword arguments for ``client.beta.messages.stream()``."""
        betas_on = self.betas
        caps = self.capabilities
        params: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens}
        if request.system:
            # Explicit breakpoint: tools + system prompt are identical on every request.
            params["system"] = [
                {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
            ]
        params["messages"] = request.messages
        if request.tools:
            tools = [dict(t) for t in request.tools]
            if betas_on:
                for t in tools:
                    t["eager_input_streaming"] = True
            params["tools"] = tools
        if betas_on:
            # Automatic caching moves a breakpoint to the end of the growing conversation.
            params["cache_control"] = {"type": "ephemeral"}
        beta_flags: list[str] = []
        if caps.adaptive_thinking:
            thinking: dict[str, Any] = {"type": "adaptive"}
            if betas_on and caps.progress_updates:
                thinking["display"] = "updates"
                beta_flags.append(_BETA_PROGRESS_UPDATES)
            params["thinking"] = thinking
        if self.effort and caps.effort:
            params["output_config"] = {"effort": self.effort}
        if betas_on and caps.fallbacks:
            params["fallbacks"] = "default"
            beta_flags.append(_BETA_FALLBACKS)
        if beta_flags:
            params["betas"] = beta_flags
        return params

    def generate(self, request: ModelRequest, emit: Callable[[Event], None]) -> ModelResponse:
        params = self.build_params(request)
        with contextlib.ExitStack() as stack:
            try:
                stream = stack.enter_context(self.client.beta.messages.stream(**params))
            except TypeError as exc:
                # The SDK raises a bare TypeError when it can't find any credentials.
                if "authentication" in str(exc):
                    raise MissingCredentials(_CREDENTIALS_HELP) from exc
                raise
            notes: dict[int, list[str]] = {}
            events = iter(stream)
            while True:
                try:
                    event = next(events)
                except StopIteration:
                    break
                except ValueError as exc:
                    # Only the SDK's tool-input JSON error is safe to retry; anything else
                    # (including typed API errors, which aren't ValueErrors) propagates.
                    if "tool parameter JSON" in str(exc):
                        raise MalformedToolInput(str(exc)) from exc
                    raise
                _dispatch(event, emit, notes)
            message = stream.get_final_message()
        return _to_response(message)


def _dispatch(event: Any, emit: Callable[[Event], None], notes: dict[int, list[str]]) -> None:
    kind = getattr(event, "type", None)
    if kind == "content_block_start":
        block = event.content_block
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            emit(ToolUseStarted(block.name))
        elif block_type == "fallback":
            emit(ModelFallback(block.from_.model, block.to.model))
        elif block_type == "thinking":
            notes[event.index] = [getattr(block, "thinking", "") or ""]
    elif kind == "content_block_delta":
        delta = event.delta
        if delta.type == "text_delta":
            emit(TextDelta(delta.text))
        elif delta.type == "thinking_delta" and event.index in notes:
            notes[event.index].append(delta.thinking)
    elif kind == "content_block_stop":
        # With display="updates", any thinking block with text is a progress note.
        text = "".join(notes.pop(event.index, [])).strip()
        if text:
            emit(ProgressNote(text))


def _to_response(message: Any) -> ModelResponse:
    usage = message.usage
    refusal = None
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        refusal = Refusal(
            category=getattr(details, "category", None),
            explanation=getattr(details, "explanation", None),
        )
    return ModelResponse(
        content=[_block_to_dict(block) for block in message.content],
        stop_reason=message.stop_reason,
        model=str(message.model),
        usage=Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        ),
        refusal=refusal,
    )


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Serialize a content block exactly the way the SDK would send it back."""
    if isinstance(block, dict):
        return dict(block)
    data = block.to_dict(mode="json")
    data.pop("parsed_output", None)
    return data
