# AgentForge

**Tool-calling AI agents in plain Python, plus a batteries-included assistant for your terminal.**

AgentForge turns typed Python functions into tools that Claude can call, and runs the agent
loop for you: streaming, parallel tool calls, approvals, path sandboxing and offline testing.
It ships with an `agentforge` command and 35 built-in tools for the things developers and
SREs do all day: reading code, digging through git history, checking ports and
certificates, running commands, and doing exact conversions that language models get wrong
when they do them in their heads (timestamps, cron, regexes, subnets, JWTs).

```console
$ git diff --staged | agentforge ask "review this for bugs and missing tests"
$ agentforge "what's listening on port 8080, and which process owns it?"
$ agentforge ask "when does the TLS certificate for example.com expire?"
```

- [Install](#install)
- [Use it from the terminal](#use-it-from-the-terminal)
- [Safety model](#safety-model)
- [Built-in tools](#built-in-tools)
- [Build your own agent](#build-your-own-agent)
- [Test agents offline](#test-agents-offline)
- [How it talks to Claude](#how-it-talks-to-claude)
- [Configuration](#configuration)
- [Development](#development)

## Install

Python 3.10 or newer:

```bash
pip install "git+https://github.com/purple-ranger-0/agentforge"
export ANTHROPIC_API_KEY=sk-ant-...      # or sign in once with `ant auth login`
```

The only runtime dependency is the official [`anthropic`](https://pypi.org/project/anthropic/) SDK.

## Use it from the terminal

```bash
agentforge                         # interactive session (same as `agentforge chat`)
agentforge "your question"         # one-shot (same as `agentforge ask "your question"`)
agentforge tools                   # list the built-in tools
```

Anything piped in is attached to the request, so it composes with the rest of your toolbox:

```bash
# Code and git
git diff --staged | agentforge ask "review this diff; point out bugs and missing tests"
agentforge ask "summarize what changed on this branch compared to main"
agentforge ask "find every TODO and FIXME and group them by who last touched the line"
agentforge ask "why is this repo 2 GB?"

# Ops and debugging
kubectl get pods -A | agentforge ask "which pods are unhealthy and why?"
tail -n 300 /var/log/app.log | agentforge ask "what went wrong around 14:02?"
agentforge ask "is anything listening on 5432, and can I reach db.internal on 5432?"
agentforge ask "which processes are using the most memory right now?"

# Exact conversions
agentforge ask "decode this JWT and tell me whether it has expired: eyJhbGciOi..."
agentforge ask "what is 1767225600000 in Asia/Kolkata?"
agentforge ask "explain */15 9-17 * * mon-fri and list the next 5 runs in Europe/Berlin"
agentforge ask "is 10.0.3.17 inside 10.0.0.0/22? split that network into /24s"

# Let it act (it asks before anything risky; --yes skips the questions)
agentforge ask "run the test suite and fix whatever fails"
```

When stdout is a terminal you see the answer stream in, with one line per tool call. When
stdout is a pipe or a file, only the final answer goes to stdout, and tool activity and the
usage summary go to stderr, so `agentforge ask ... > answer.md` does what you'd expect.
`--json` prints the answer, stop reason, token usage and every tool call as JSON.

In `chat`, use `/help`, `/tools`, `/usage`, `/save [file]`, `/clear` and `/exit`. End a line
with `\` to keep typing on the next one, and press Ctrl-C to interrupt a request.

## Safety model

Every tool has a risk level, and the risk level decides whether it runs on its own:

| Risk | Examples | Default (`--mode ask`) |
|---|---|---|
| `read` | `read_file`, `git_log`, `list_processes`, `calculate` | runs |
| `write` | `write_file`, `edit_file` | asks first |
| `network` | `fetch_url`, `check_ports`, `tls_certificate` | asks first |
| `exec` | `run_command`, `http_request` (POST/PUT/DELETE...) | asks first |

- **Approvals show exactly what will happen:** the diff an edit will make, the full command
  line, or the HTTP request. Answer `y`, `n`, or `a` to always allow that tool for the rest
  of the session.
- **Modes:** `--mode ask` (default), `--mode auto` or `--yes` (never ask), and `--mode readonly`
  (risky tools aren't offered to the model at all).
- **Fine-grained rules:** `--allow network` or `--allow edit_file` auto-approves a risk level or
  a tool; `--deny exec` hides one completely.
- **No terminal, no risky calls:** without a terminal to ask on (for example in CI), risky
  calls are refused and the model is told how the user can allow them.
- **Workspace confinement:** file tools only reach paths under the workspace root (the
  current directory, or `--root DIR`). Symlinks that point outside are refused too.
- **Hardened tools:** shell commands run with stdin closed, a timeout, and the whole process
  group killed on timeout. Git runs without fsmonitor hooks, external diff tools or
  textconv, and revision arguments can't be smuggled in as options. HTTP is limited to
  http(s), and cookie values are redacted. `regex_test` runs patterns in a separate, killable
  process, so a catastrophically backtracking pattern can't hang the agent.

## Built-in tools

| Pack | Tools |
|---|---|
| `fs` | `read_file` (paging, tail of logs) · `list_dir` · `find_files` (globs, respects .gitignore) · `search_text` (regex grep with context) · `file_info` · `disk_usage` · `diff_files` · `write_file` ✎ · `edit_file` ✎ (exact replacement, diff preview) |
| `git` | `git_status` · `git_log` (filters by path, author, message, date) · `git_diff` (working tree, staged, branch ranges) · `git_show` (commit, stat, or file at a revision) · `git_blame` · `git_branches` |
| `shell` | `run_command` ⚡ |
| `net` | `fetch_url` 🌐 (status, headers, timing, JSON pretty-printing, HTML to text) · `http_request` ⚡ · `dns_lookup` 🌐 · `check_ports` 🌐 · `tls_certificate` 🌐 (expiry, SANs, verification errors) |
| `system` | `system_info` · `list_processes` · `listening_ports` · `which` |
| `data` | `decode_jwt` · `convert_encoding` (base64/base64url/hex/URL/HTML) · `compute_hash` · `convert_time` (epochs, ISO 8601, time zones) · `cron_schedule` · `regex_test` · `calculate` · `cidr_info` · `generate_secret` · `json_query` |

✎ write · 🌐 network · ⚡ exec. Everything else is read-only. Choose packs with
`--tools fs,git` (pack names and tool names both work). `agentforge tools --json` prints the
full definitions.

## Build your own agent

Any typed Python function becomes a tool. The signature becomes the JSON Schema, and the
docstring becomes the description (`Args:` entries describe the parameters):

```python
from typing import Literal

from agentforge import Agent, ToolError, tool


@tool
def ticket_status(ticket_id: str) -> dict:
    """Look up a support ticket. Use it whenever the user mentions a ticket key.

    Args:
        ticket_id: The ticket key, e.g. "OPS-123".
    """
    if not ticket_id.startswith("OPS-"):
        raise ToolError("ticket keys look like OPS-123")  # sent back to the model as an error
    return {"id": ticket_id, "status": "open", "assignee": "dana"}


@tool(risk="exec")
def restart(service: Literal["api", "worker"], reason: str) -> str:
    """Restart one of our services. Only when the user explicitly asks."""
    ...


agent = Agent([ticket_status, restart], system="You are our on-call assistant.")
print(agent.run("What's going on with OPS-123?").text)
```

- **Types:** `str`, `int`, `float`, `bool`, `Path`, `list[...]`, `dict[str, ...]`, `Literal[...]`,
  `Enum`s, `X | None`, unions and `Annotated[X, "description"]`. Arguments from the model
  are validated and converted before your function runs. A bad call never reaches your
  code; the model gets a precise error to correct instead.
- **Errors:** raise `ToolError("...")` to tell the model what went wrong. Any other exception
  is reported to the model as `ExceptionType: message` instead of crashing the agent.
- **Return values:** strings are passed through; dicts, lists, numbers and dataclasses are
  sent as JSON. Very long results are trimmed, keeping the head and the tail.
- **Context:** add a parameter annotated `ToolContext` to get the workspace (`ctx.workspace.resolve(path)`
  enforces the sandbox) and a per-session `ctx.state` dict. It's injected, not shown to the model.
- **Async:** `async def` tools work too.
- **Risk and previews:** `@tool(risk="write", preview=fn)`: `fn(ctx, args)` returns the text shown
  in the approval prompt. A preview can raise `ToolError` to reject a call that can't
  work, without bothering the user.
- **Decorated functions stay callable**, so you can unit-test them directly.

Mix in built-in packs, follow along with events, and keep a conversation going:

```python
from agentforge import Agent, AnthropicModel, PermissionPolicy, events
from agentforge.toolkit import load


def show(event):
    if isinstance(event, events.TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, events.ToolCallStarted):
        print(f"\n-> {event.call.name}({event.call.args})")


agent = Agent(
    [*load(["fs", "git"]), ticket_status],
    model=AnthropicModel("claude-opus-5-5", effort="high"),
    policy=PermissionPolicy("readonly"),
    workspace="~/src/billing",
    on_event=show,
)
session = agent.session()
session.send("Which commits touched the invoice code last week?")
session.send("Who should review a change there?")
session.save("transcript.json")
```

`PermissionPolicy(mode, allow=..., deny=..., approver=fn)` lets your app decide on risky calls.
`fn(tool, call)` returns `Approval.ALLOW`, `Approval.DENY` or `Approval.ALWAYS`, and
`call.preview` holds the preview text. Events include `TextDelta`, `ProgressNote`,
`ToolUseStarted`, `ToolCallStarted`, `ToolCallFinished`, `ToolCallRejected`, `ModelFallback`,
`TurnStarted` and `TurnFinished`.

Your tools work in the CLI too:

```bash
agentforge chat --load examples/team_tools.py
```

## Test agents offline

`ScriptedModel` plays the model's side of the conversation, so you can test what your agent
does with each tool call without API keys, cost or flakiness:

```python
from agentforge import Agent
from agentforge.testing import ScriptedModel, call, reply, say


def test_ticket_lookup():
    model = ScriptedModel(
        [
            reply("Let me check.", call("ticket_status", ticket_id="OPS-123")),
            say("OPS-123 is open and assigned to Dana."),
        ]
    )
    result = Agent([ticket_status], model=model).run("status of OPS-123?")

    assert result.text == "OPS-123 is open and assigned to Dana."
    assert result.tool_calls[0].args == {"ticket_id": "OPS-123"}
    assert '"assignee": "dana"' in result.tool_calls[0].result.content
    # every request the agent sent is recorded, so you can assert on prompts and history too
    assert model.requests[1].messages[-1]["content"][0]["type"] == "tool_result"
```

A step can also be a function that receives the request and returns a reply, which is useful
for assertions mid-conversation. There's also `refusal()`. See
[`examples/test_team_tools.py`](examples/test_team_tools.py).

## How it talks to Claude

- **Model:** `claude-opus-5-5` by default (`--model` to change), with adaptive thinking and an
  explicit `effort` (`--effort`, default `medium`). Requests stream, with room for long
  agentic turns (`max_tokens` 64k).
- **Prompt caching:** the tool definitions and system prompt are kept byte-for-byte stable,
  and per-session details (workspace, date, OS) go into the first message instead. A cache
  breakpoint sits on the system prompt, and automatic caching covers the growing
  conversation, so follow-up turns read most of the prompt from cache. The usage line shows
  cached tokens and an estimated cost.
- **Append-only history:** assistant turns, including thinking blocks, are sent back
  unchanged, and the tool set is fixed for a session. This keeps the cache warm and
  is what newer models require for their reasoning to carry across turns.
- **Parallel tool calls:** read-only and network calls in one turn run concurrently.
  Side-effecting calls run one at a time, in order. All results return in a single message.
- **Refusal fallback:** with `fallbacks: "default"`, if a safety classifier declines a
  request, the API continues on Anthropic's recommended fallback model instead of failing the
  turn. AgentForge drops the declined attempt's partial tool calls, as the API requires.
- **Progress notes:** `thinking.display: "updates"` returns the short notes Claude writes
  between tool calls, so long tasks don't look frozen.
- **Streaming tool inputs:** large tool inputs, like file contents, stream as they're generated
  (`eager_input_streaming`), and every input is validated before anything runs.
- **Gateways and proxies:** refusal fallback, progress notes, streamed tool inputs and the
  automatic conversation cache breakpoint use request fields that a gateway may reject.
  They switch off automatically when `ANTHROPIC_BASE_URL` points somewhere other than
  the first-party API, and `--compat` (or `AnthropicModel(betas=False)`) turns them off
  explicitly.

## Configuration

| Flag | Environment | Default | |
|---|---|---|---|
| `-m, --model` | `AGENTFORGE_MODEL` | `claude-opus-5-5` | Claude model ID |
| `-e, --effort` | `AGENTFORGE_EFFORT` | `medium` | `low`, `medium`, `high`, `xhigh`, `max` |
| `-t, --tools` | `AGENTFORGE_TOOLS` | `all` | packs and/or tool names, comma-separated |
| `--load FILE` | | | add `@tool` functions from a Python file (repeatable) |
| `-C, --root DIR` | | `.` | workspace root for file tools |
| `--mode`, `-y/--yes` | | `ask` | `ask`, `auto`, `readonly` |
| `--allow`, `--deny` | | | tool names or risk levels (repeatable, comma-separated) |
| `--max-turns` | | `50` | model turns per request |
| `--max-tokens` | | `64000` | output limit per model turn |
| `--save FILE` | | | write the transcript as JSON |
| `--compat` | | | don't send beta request fields |
| `-v, --verbose` | | | show tool output and debug logs |
| `--no-color` | `NO_COLOR` | | |

Credentials are resolved by the Anthropic SDK (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or
an `ant auth login` profile). Exit codes: `0` success, `1` API or configuration error, `2`
usage error, `3` incomplete (refusal, turn limit, output limit), `130` interrupted.

## Development

```bash
git clone https://github.com/purple-ranger-0/agentforge && cd agentforge
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest tests examples      # fully offline, no API key needed
ruff check . && ruff format --check .
```

```
src/agentforge/
  tools.py, schema.py     @tool, JSON Schema generation, input validation
  agent.py                the agent loop (Agent, Session, RunResult)
  model.py                AnthropicModel and the ModelBackend protocol
  policy.py, workspace.py approvals and path confinement
  testing.py              ScriptedModel and helpers
  toolkit/                the built-in tool packs
  cli.py, render.py       the agentforge command
```

The test suite runs every built-in tool for real (temporary git repositories, local HTTP and
TLS servers, real subprocesses). It covers the agent loop with scripted models, and the
Claude integration through the real SDK against mocked HTTP streams. CI runs it on Python
3.10 to 3.13 on Linux. macOS should work (the tools fall back to `lsof`, `sysctl` and BSD
`ps`) but isn't tested in CI yet.
