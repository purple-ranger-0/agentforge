"""Embed an agent in your own program: built-in tools plus your own, with live events.

export ANTHROPIC_API_KEY=...
python examples/embed.py "Which files changed in the last 5 commits, and why?"
"""

from __future__ import annotations

import sys

from agentforge import Agent, AnthropicModel, PermissionPolicy, events, tool
from agentforge.toolkit import load


@tool
def count_lines(text: str) -> int:
    """Count the lines in a piece of text."""
    return len(text.splitlines())


def show(event: events.Event) -> None:
    """Print the answer as it streams, plus one line per tool call."""
    if isinstance(event, events.TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, events.ProgressNote):
        print(f"\n  ({event.text})", flush=True)
    elif isinstance(event, events.ToolCallStarted):
        print(f"\n  -> {event.call.name} {event.call.args}", flush=True)
    elif isinstance(event, events.ToolCallFinished) and event.result.is_error:
        print(f"     failed: {event.result.content}", flush=True)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Summarize the last 5 commits in this repository."
    agent = Agent(
        [*load(["fs", "git"]), count_lines],
        model=AnthropicModel(effort="low"),  # quick lookups don't need deep thinking
        policy=PermissionPolicy("readonly"),  # this agent can look, but not touch
        on_event=show,
    )
    session = agent.session()
    result = session.send(question)
    print(f"\n\n[{result.turns} turns, {len(result.tool_calls)} tool calls]")

    follow_up = session.send("Which of those changes looks riskiest, in one sentence?")
    used = session.usage.output_tokens
    print(f"\n\n[{used} output tokens in total; finished with {follow_up.stop_reason}]")


if __name__ == "__main__":
    main()
