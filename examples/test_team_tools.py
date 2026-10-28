"""Offline tests for an agent built on team_tools.py. Run with `pytest examples`.

ScriptedModel plays the model's side of the conversation, so these tests are fast,
deterministic and free, and they check what your code does with each tool call.
"""

from __future__ import annotations

from team_tools import page_on_call, service_info, who_is_on_call

from agentforge import Agent, Approval, PermissionPolicy
from agentforge.testing import ScriptedModel, call, say

TOOLS = [who_is_on_call, service_info, page_on_call]


def test_escalation_flow() -> None:
    model = ScriptedModel(
        [
            call("service_info", name="checkout-api"),
            call("who_is_on_call", team="payments"),
            say("Priya Raman is on call for checkout-api; ping #payments-oncall."),
        ]
    )
    result = Agent(TOOLS, model=model).run("checkout-api is throwing 500s. Who do I call?")

    assert [c.name for c in result.tool_calls] == ["service_info", "who_is_on_call"]
    assert '"primary": "Priya Raman"' in result.tool_calls[1].result.content
    assert result.text.startswith("Priya Raman")


def test_paging_needs_approval_and_shows_a_preview() -> None:
    previews = []

    def approver(tool, call):
        previews.append(call.preview)
        return Approval.DENY

    model = ScriptedModel(
        [
            call("page_on_call", team="data", message="ingest lag is over 30 minutes"),
            say("I didn't page anyone."),
        ]
    )
    agent = Agent(TOOLS, model=model, policy=PermissionPolicy(approver=approver))
    result = agent.run("Page the data team about ingest lag.")

    assert previews == ["Page Marta Silva (data primary):\n  ingest lag is over 30 minutes"]
    assert result.tool_calls[0].result.is_error
    tool_result = model.requests[1].messages[-1]["content"][0]
    assert tool_result["content"].startswith("Permission denied")


def test_bad_arguments_are_explained_to_the_model() -> None:
    model = ScriptedModel(
        [call("who_is_on_call", team="marketing"), call("service_info", name="nope"), say("?")]
    )
    result = Agent(TOOLS, model=model).run("Who's on call for marketing?")

    assert 'expected one of "platform", "payments", "data"' in result.tool_calls[0].result.content
    assert "known services: checkout-api" in result.tool_calls[1].result.content
