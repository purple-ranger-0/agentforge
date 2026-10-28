"""Example custom tools: try `agentforge chat --load examples/team_tools.py`.

The data is made up; swap the function bodies for calls to your real systems (paging,
service catalog, wiki, ...). The model decides when to call each tool from its docstring,
so say *when* a tool is useful, not just what it does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from agentforge import ToolContext, ToolError, tool

Team = Literal["platform", "payments", "data"]

ROTA: dict[str, dict[str, str]] = {
    "platform": {"primary": "Dana Ito", "secondary": "Sam Okafor", "channel": "#platform-oncall"},
    "payments": {"primary": "Priya Raman", "secondary": "Lee Novak", "channel": "#payments-oncall"},
    "data": {"primary": "Marta Silva", "secondary": "Chris Wu", "channel": "#data-oncall"},
}

SERVICES: dict[str, dict[str, Any]] = {
    "checkout-api": {"team": "payments", "runbook": "runbooks/checkout-api.md", "port": 8443},
    "ingest-worker": {"team": "data", "runbook": "runbooks/ingest-worker.md", "port": 9100},
    "edge-proxy": {"team": "platform", "runbook": "runbooks/edge-proxy.md", "port": 443},
}


@tool
def who_is_on_call(team: Team) -> dict[str, str]:
    """Look up who is on call for a team right now and where to reach them.

    Use it whenever the user asks who to contact or escalate to.

    Args:
        team: The owning team.
    """
    return ROTA[team]


@tool
def service_info(name: str) -> dict[str, Any]:
    """Look up a service in the service catalog: owning team, runbook path and port.

    Use it to find out who owns a service or where its runbook is.

    Args:
        name: Service name, e.g. "checkout-api".
    """
    if name not in SERVICES:
        raise ToolError(f"unknown service {name!r}; known services: {', '.join(SERVICES)}")
    return {"name": name, **SERVICES[name]}


def _page_preview(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    person = ROTA[args["team"]]["primary"]
    return f"Page {person} ({args['team']} primary):\n  {args['message']}"


@tool(risk="exec", preview=_page_preview)
def page_on_call(team: Team, message: str) -> str:
    """Page a team's primary on-call engineer. Only do this when the user asks you to.

    Args:
        team: The team to page.
        message: What's wrong, in one or two sentences.
    """
    person = ROTA[team]["primary"]
    # A real version would call your paging provider's API here.
    return f"Paged {person} ({team}): {message!r}"
