"""AnthropicModel against the real SDK, with HTTP served by httpx2.MockTransport."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from agentforge import AnthropicModel, MalformedToolInput, MissingCredentials, ModelRequest
from agentforge import events as ev
from agentforge.model import Usage, capabilities
from agentforge.pricing import estimate_cost

TOOLS = [
    {"name": "add", "description": "Add.", "input_schema": {"type": "object", "properties": {}}}
]
REQUEST = ModelRequest(
    system="You are a test.", tools=TOOLS, messages=[{"role": "user", "content": "hi"}]
)


def sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def message_start(model: str = "claude-opus-5-5") -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 1,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 40,
            },
        },
    }


def block(
    index: int, content_block: dict[str, Any], *deltas: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {"type": "content_block_start", "index": index, "content_block": content_block},
        *({"type": "content_block_delta", "index": index, "delta": d} for d in deltas),
        {"type": "content_block_stop", "index": index},
    ]


def finish(stop_reason: str, stop_details: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    return [
        {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 50}},
        {"type": "message_stop"},
    ]


class FakeAPI:
    """Records requests and replies with a canned SSE stream."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx2.Response(
                self.status,
                json={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "bad thing"},
                },
            )
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=self.body
        )

    @property
    def json(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def client_for(
    handler: Callable[[httpx2.Request], httpx2.Response], **kwargs: Any
) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        **kwargs,
    )


def test_request_shape_with_betas() -> None:
    api = FakeAPI(
        sse(
            message_start(),
            *block(0, {"type": "text", "text": ""}, {"type": "text_delta", "text": "hi"}),
            *finish("end_turn"),
        )
    )
    AnthropicModel(client=client_for(api)).generate(REQUEST, lambda e: None)
    body, headers = api.json, api.requests[-1].headers
    assert (
        headers["anthropic-beta"]
        == "thinking-display-updates-2026-08-18,server-side-fallback-2026-07-01"
    )
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 64000
    assert body["stream"] is True
    assert body["thinking"] == {"type": "adaptive", "display": "updates"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["system"] == [
        {"type": "text", "text": "You are a test.", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["tools"] == [{**TOOLS[0], "eager_input_streaming": True}]
    for removed in ("temperature", "top_p", "top_k", "tool_choice"):
        assert removed not in body


def test_compat_mode_sends_only_stable_fields() -> None:
    api = FakeAPI(sse(message_start(), *finish("end_turn")))
    AnthropicModel(client=client_for(api), betas=False, effort="high").generate(
        REQUEST, lambda e: None
    )
    body = api.json
    assert "anthropic-beta" not in api.requests[-1].headers
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    for absent in ("fallbacks", "cache_control"):
        assert absent not in body
    assert "eager_input_streaming" not in body["tools"][0]


def test_betas_default_off_for_a_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert AnthropicModel(client=client_for(FakeAPI(b""))).betas is True
    gateway = client_for(FakeAPI(b""), base_url="https://llm-gateway.internal")
    assert AnthropicModel(client=gateway).betas is False
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm-gateway.internal")
    assert AnthropicModel().betas is False
    assert AnthropicModel(betas=True).betas is True
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    assert AnthropicModel().betas is True


def test_streaming_events_and_response() -> None:
    body = sse(
        message_start(),
        *block(
            0,
            {"type": "thinking", "thinking": "", "signature": ""},
            {"type": "thinking_delta", "thinking": "Reading the config next."},
            {"type": "signature_delta", "signature": "sig-1"},
        ),
        *block(
            1,
            {"type": "thinking", "thinking": "", "signature": ""},
            {"type": "signature_delta", "signature": "sig-2"},
        ),
        *block(
            2,
            {"type": "text", "text": ""},
            {"type": "text_delta", "text": "Let me "},
            {"type": "text_delta", "text": "check."},
        ),
        *block(
            3,
            {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"a": 2, '},
            {"type": "input_json_delta", "partial_json": '"b": 3}'},
        ),
        *finish("tool_use"),
    )
    seen: list[Any] = []
    response = AnthropicModel(client=client_for(FakeAPI(body))).generate(REQUEST, seen.append)
    assert seen == [
        ev.ProgressNote("Reading the config next."),
        ev.TextDelta("Let me "),
        ev.TextDelta("check."),
        ev.ToolUseStarted("add"),
    ]
    assert response.stop_reason == "tool_use"
    assert response.model == "claude-opus-5-5"
    assert response.content == [
        {"type": "thinking", "thinking": "Reading the config next.", "signature": "sig-1"},
        {"type": "thinking", "thinking": "", "signature": "sig-2"},
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2, "b": 3}},
    ]
    assert response.usage == Usage(12, 50, 300, 40)
    json.dumps(response.content)  # plain, serializable dicts


def test_fallback_blocks_are_reported() -> None:
    body = sse(
        message_start(),
        *block(
            0,
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-5"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
        ),
        *block(1, {"type": "text", "text": ""}, {"type": "text_delta", "text": "Here you go."}),
        *finish("end_turn"),
    )
    seen: list[Any] = []
    response = AnthropicModel(client=client_for(FakeAPI(body))).generate(REQUEST, seen.append)
    assert seen[0] == ev.ModelFallback("claude-opus-5-5", "claude-opus-5")
    assert response.content[0]["type"] == "fallback"
    assert response.content[0]["from"] == {"model": "claude-opus-5-5"}
    assert response.model == "claude-opus-5"


def test_refusals_carry_details() -> None:
    body = sse(
        message_start(),
        *finish("refusal", {"type": "refusal", "category": "cyber", "explanation": "Not this."}),
    )
    response = AnthropicModel(client=client_for(FakeAPI(body))).generate(REQUEST, lambda e: None)
    assert response.stop_reason == "refusal"
    assert response.refusal is not None
    assert (response.refusal.category, response.refusal.explanation) == ("cyber", "Not this.")
    assert response.content == []


def test_unparseable_tool_json_raises_a_retryable_error() -> None:
    body = sse(
        message_start(),
        *block(
            0,
            {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"a": @@@'},
        ),
        *finish("tool_use"),
    )
    with pytest.raises(MalformedToolInput):
        AnthropicModel(client=client_for(FakeAPI(body))).generate(REQUEST, lambda e: None)


def test_api_errors_propagate_as_typed_sdk_errors() -> None:
    with pytest.raises(anthropic.BadRequestError):
        AnthropicModel(client=client_for(FakeAPI(b"", status=400))).generate(
            REQUEST, lambda e: None
        )


def test_missing_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    client = anthropic.Anthropic(
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(FakeAPI(b"")))
    )
    with pytest.raises(MissingCredentials, match="ANTHROPIC_API_KEY"):
        AnthropicModel(client=client).generate(REQUEST, lambda e: None)


@pytest.mark.parametrize(
    ("model", "adaptive", "progress", "fallbacks"),
    [
        ("claude-opus-5-5", True, True, True),
        ("claude-sonnet-5-5", True, True, True),
        ("claude-fable-5-1", True, True, True),
        ("claude-opus-5", True, False, True),
        ("claude-haiku-5-5", True, False, False),
        ("claude-sonnet-4-6", True, False, False),
        ("anthropic.claude-opus-5-5", True, True, True),
        ("claude-future-9", True, False, False),
        ("claude-haiku-4-5", False, False, False),
        ("claude-sonnet-4-20250514", False, False, False),
        ("claude-3-7-sonnet-latest", False, False, False),
    ],
)
def test_capabilities(model: str, adaptive: bool, progress: bool, fallbacks: bool) -> None:
    caps = capabilities(model)
    assert (caps.adaptive_thinking, caps.effort, caps.progress_updates, caps.fallbacks) == (
        adaptive,
        adaptive,
        progress,
        fallbacks,
    )


def test_legacy_models_get_no_thinking_or_effort() -> None:
    params = AnthropicModel("claude-haiku-4-5", betas=True).build_params(REQUEST)
    assert "thinking" not in params and "output_config" not in params and "fallbacks" not in params
    assert "betas" not in params


def test_effort_is_validated() -> None:
    with pytest.raises(ValueError, match="effort must be one of"):
        AnthropicModel(effort="extreme")
    assert "output_config" not in AnthropicModel(effort=None, betas=False).build_params(REQUEST)


def test_cost_estimates() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    assert estimate_cost("claude-opus-5-5", usage) == pytest.approx(4 + 20 + 0.2 + 5)
    assert estimate_cost("some-other-model", usage) is None
    assert Usage(1, 2, 3, 4).prompt_tokens == 8
    assert (Usage(1, 1, 1, 1) + Usage(1, 2, 3, 4)) == Usage(2, 3, 4, 5)
