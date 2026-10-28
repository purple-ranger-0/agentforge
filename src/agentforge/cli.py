"""The ``agentforge`` command: a tool-calling assistant for your terminal."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import logging
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from . import toolkit
from ._version import __version__
from .agent import Agent, RunResult, Session
from .errors import AgentForgeError
from .model import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    EFFORT_LEVELS,
    AnthropicModel,
    ModelBackend,
)
from .policy import MODES, PermissionPolicy
from .render import Renderer, Terminal, color_enabled, isatty, usage_summary
from .tools import Tool
from .workspace import Workspace

SUBCOMMANDS = ("chat", "ask", "tools", "version")
MAX_STDIN_CHARS = 1_000_000

DESCRIPTION = """\
AgentForge: a tool-calling AI assistant for your terminal. It reads and searches your
code, digs through git history, checks your machine and network, runs commands (with your
approval), and does exact conversions: JWTs, timestamps, cron, regexes, subnets and more."""

EPILOG = """\
examples:
  agentforge                                        interactive session
  agentforge "what's listening on port 8080?"       one-shot question
  git diff --staged | agentforge ask "review this for bugs"
  kubectl get pods -A | agentforge ask "which pods are unhealthy and why?"
  agentforge ask --yes "run the tests and fix what fails"
  agentforge ask -t data "when does this JWT expire? eyJ..."
  agentforge chat --load ./my_tools.py              add your own @tool functions
  agentforge tools                                  list the built-in tools

Needs ANTHROPIC_API_KEY (or `ant auth login`). Defaults can be set with AGENTFORGE_MODEL,
AGENTFORGE_EFFORT and AGENTFORGE_TOOLS."""


class UsageError(Exception):
    """A problem with how the command was invoked."""


class _Streams:
    def __init__(self, stdin: TextIO, out: TextIO, err: TextIO) -> None:
        self.stdin, self.out, self.err = stdin, out, err


def main(
    argv: list[str] | None = None,
    *,
    model: ModelBackend | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI. ``model`` and the streams can be injected for testing."""
    io = _Streams(stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr)
    args = build_parser().parse_args(route(list(sys.argv[1:] if argv is None else argv)))
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
        logging.getLogger("agentforge").setLevel(logging.DEBUG)
    try:
        if args.command == "version":
            io.out.write(f"agentforge {__version__}\n")
            return 0
        if args.command == "tools":
            return cmd_tools(args, io)
        if args.command == "ask":
            return cmd_ask(args, io, model)
        return cmd_chat(args, io, model)
    except UsageError as exc:
        io.err.write(f"agentforge: {exc}\n")
        return 2
    except KeyboardInterrupt:
        io.err.write("\n")
        return 130
    except BrokenPipeError:
        # The reader went away (e.g. `agentforge tools | head`). Point stdout at devnull so
        # the interpreter's final flush doesn't raise again, and exit like SIGPIPE would.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), io.out.fileno())
        except (AttributeError, OSError, ValueError):
            pass
        return 141


def route(argv: list[str]) -> list[str]:
    """``agentforge`` alone starts chat; ``agentforge "question"`` means ask."""
    if not argv:
        return ["chat"]
    first = argv[0]
    if first in SUBCOMMANDS or first in ("-h", "--help", "-V", "--version"):
        return argv
    if first.startswith("-"):
        return ["chat", *argv]
    return ["ask", *argv]


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("agent options")
    group.add_argument(
        "-m",
        "--model",
        default=os.environ.get("AGENTFORGE_MODEL", DEFAULT_MODEL),
        help=f"Claude model ID (default: {DEFAULT_MODEL})",
    )
    group.add_argument(
        "-e",
        "--effort",
        choices=EFFORT_LEVELS,
        default=os.environ.get("AGENTFORGE_EFFORT", DEFAULT_EFFORT),
        help=f"how hard the model thinks (default: {DEFAULT_EFFORT})",
    )
    group.add_argument(
        "-t",
        "--tools",
        default=os.environ.get("AGENTFORGE_TOOLS", "all"),
        metavar="PACKS",
        help="tool packs and/or tool names, comma-separated: fs, git, shell, net, system, "
        "data, or all (default)",
    )
    group.add_argument(
        "--load",
        action="append",
        default=[],
        metavar="FILE",
        help="add the @tool functions defined in a Python file (repeatable)",
    )
    group.add_argument(
        "-C",
        "--root",
        default=".",
        metavar="DIR",
        help="workspace root that file tools are confined to (default: current directory)",
    )
    group.add_argument(
        "--mode",
        choices=MODES,
        default="ask",
        help="ask: confirm risky tools (default); auto: never ask; readonly: read-only tools only",
    )
    group.add_argument(
        "-y", "--yes", action="store_true", help="run every tool call without asking (--mode auto)"
    )
    group.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="NAME",
        help="auto-approve a tool or a risk level (write, network, exec); repeatable",
    )
    group.add_argument(
        "--deny",
        action="append",
        default=[],
        metavar="NAME",
        help="never offer a tool or risk level to the model; repeatable",
    )
    group.add_argument(
        "--max-turns", type=int, default=50, help="model turns per request (default: 50)"
    )
    group.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"output token limit per model turn (default: {DEFAULT_MAX_TOKENS})",
    )
    group.add_argument(
        "--compat",
        action="store_true",
        help="don't send beta request features (use with gateways and proxies)",
    )
    group.add_argument(
        "--save", metavar="FILE", help="write the conversation transcript (JSON) to FILE"
    )
    group.add_argument(
        "-v", "--verbose", action="store_true", help="show tool output and debug logs"
    )
    group.add_argument("--no-color", action="store_true", help="disable colors (or set NO_COLOR)")

    parser = argparse.ArgumentParser(
        prog="agentforge",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"agentforge {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.add_parser("chat", parents=[common], help="interactive session (the default)")
    ask = sub.add_parser(
        "ask", parents=[common], help="answer one request; text piped on stdin is included"
    )
    ask.add_argument("prompt", nargs="*", help="the request")
    ask.add_argument("--json", action="store_true", help="print the result as JSON")
    tools = sub.add_parser("tools", help="list the available tools")
    tools.add_argument("-t", "--tools", default="all", metavar="PACKS", help="packs or tool names")
    tools.add_argument("--load", action="append", default=[], metavar="FILE")
    fmt = tools.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="full definitions as JSON")
    fmt.add_argument("--markdown", action="store_true", help="a Markdown table")
    sub.add_parser("version", help="print the version")
    return parser


# --------------------------------------------------------------------------- tools


def split_names(values: Iterable[str]) -> list[str]:
    return [name.strip() for value in values for name in value.split(",") if name.strip()]


def load_tool_file(path: str) -> list[Tool]:
    """Import a Python file and collect the Tool objects it defines."""
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise UsageError(f"--load: no such file: {path}")
    module_name = f"agentforge_user_tools_{file.stem}_{abs(hash(str(file)))}"
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:
        raise UsageError(f"--load: can't import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[module_name]
        raise UsageError(f"--load {path}: {type(exc).__name__}: {exc}") from exc
    tools = [value for value in vars(module).values() if isinstance(value, Tool)]
    if not tools:
        raise UsageError(f"--load {path}: no @tool functions found")
    return tools


def resolve_tools(spec: str, files: Iterable[str]) -> list[Tool]:
    try:
        tools = {t.name: t for t in toolkit.load(split_names([spec]))}
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    for path in files:
        tools.update((t.name, t) for t in load_tool_file(path))  # custom tools win on a clash
    return list(tools.values())


def cmd_tools(args: argparse.Namespace, io: _Streams) -> int:
    tools = resolve_tools(args.tools, args.load)
    if args.json:
        data = [
            {**t.to_param(), "pack": toolkit.pack_of(t) or "custom", "risk": t.risk.value}
            for t in tools
        ]
        io.out.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return 0
    if args.markdown:
        io.out.write("| Tool | Pack | Risk | What it does |\n|---|---|---|---|\n")
        for t in tools:
            summary = first_sentence(t.description, limit=1000).replace("|", "\\|")
            io.out.write(
                f"| `{t.name}` | {toolkit.pack_of(t) or 'custom'} | {t.risk.value} | {summary} |\n"
            )
        return 0
    style_on = color_enabled(io.out)
    by_pack: dict[str, list[Tool]] = {}
    for t in tools:
        by_pack.setdefault(toolkit.pack_of(t) or "custom", []).append(t)
    for pack, members in by_pack.items():
        header = f"{pack} ({len(members)})"
        io.out.write((f"\033[1m{header}\033[0m" if style_on else header) + "\n")
        for t in members:
            io.out.write(f"  {t.name:<17} {t.risk.value:<8} {first_sentence(t.description)}\n")
    io.out.write(
        "\nRisk levels: read runs freely; write, network and exec ask first "
        "(see --mode and --allow).\n"
    )
    return 0


_ABBREVIATIONS = ("e.g", "i.e", "etc", "vs")


def first_sentence(text: str, limit: int = 110) -> str:
    """The first sentence of a description (ignoring "e.g." and friends), shortened to fit."""
    flat = " ".join(text.split())
    sentence = flat
    for match in re.finditer(r"\.(?:\s|$)", flat):
        if not flat[: match.start()].lower().endswith(_ABBREVIATIONS):
            sentence = flat[: match.start() + 1]
            break
    return sentence if len(sentence) <= limit else sentence[: limit - 3].rstrip() + "..."


# ---------------------------------------------------------------------------- agent


def make_renderer(args: argparse.Namespace, io: _Streams, *, json_mode: bool = False) -> Renderer:
    if isatty(io.out) and not json_mode:
        on = color_enabled(io.out, args.no_color)
        return Renderer(
            io.out,
            io.out,
            terminal=Terminal(io.stdin, io.out),
            color=on,
            status_color=on,
            verbose=args.verbose,
        )
    # stdout is a pipe or file (or --json): only the answer goes there, the rest to stderr.
    return Renderer(
        io.out,
        io.err,
        terminal=Terminal(io.stdin, io.err),
        stream_text=False,
        print_answer=not json_mode,
        status_color=color_enabled(io.err, args.no_color),
        verbose=args.verbose,
    )


def build_agent(
    args: argparse.Namespace, renderer: Renderer, model: ModelBackend | None = None
) -> Agent:
    tools = resolve_tools(args.tools, args.load)
    mode = "auto" if args.yes else args.mode
    terminal = renderer.terminal
    can_ask = terminal is not None and terminal.available()
    policy = PermissionPolicy(
        mode,
        allow=split_names(args.allow),
        deny=split_names(args.deny),
        approver=renderer.approve if can_ask else None,
    )
    if model is None:
        model = AnthropicModel(
            args.model,
            effort=args.effort,
            max_tokens=args.max_tokens,
            betas=False if args.compat else None,
        )
    try:
        workspace = Workspace(args.root)
    except NotADirectoryError as exc:
        raise UsageError(str(exc)) from exc
    return Agent(
        tools,
        model=model,
        policy=policy,
        workspace=workspace,
        max_turns=args.max_turns,
        on_event=renderer,
    )


def describe_error(exc: BaseException, model: str) -> str | None:
    """A one-line explanation for errors worth reporting without a traceback."""
    if isinstance(exc, AgentForgeError):
        return str(exc)
    try:
        import anthropic
    except ImportError:  # pragma: no cover - anthropic is a dependency
        return None
    if isinstance(exc, anthropic.AuthenticationError):
        return "authentication failed; check ANTHROPIC_API_KEY"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return f"permission denied: {exc.message}"
    if isinstance(exc, anthropic.NotFoundError):
        return f"not found: {exc.message} (is {model!r} a valid model ID?)"
    if isinstance(exc, anthropic.RateLimitError):
        return "rate limited by the API; wait a moment and try again"
    if isinstance(exc, anthropic.BadRequestError):
        return (
            f"the API rejected the request: {exc.message} (behind a proxy or gateway? try --compat)"
        )
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error {exc.status_code}: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return f"couldn't reach the Anthropic API: {exc}"
    return None


def run_request(session: Session, prompt: str, renderer: Renderer, model: str) -> RunResult | None:
    """Send one request; report known errors and return None for them."""
    try:
        import anthropic

        known: tuple[type[BaseException], ...] = (AgentForgeError, anthropic.APIError)
    except ImportError:  # pragma: no cover
        known = (AgentForgeError,)
    try:
        return session.send(prompt)
    except known as exc:
        message = describe_error(exc, model) or f"{type(exc).__name__}: {exc}"
        renderer.status_line(renderer.sstyle.red(f"error: {message}"))
        return None


def exit_code(result: RunResult) -> int:
    if result.completed:
        return 0
    return 130 if result.stop_reason == "interrupted" else 3


def attach_stdin(prompt: str, piped: str, io: _Streams) -> str:
    text = piped
    if len(text) > MAX_STDIN_CHARS:
        half = MAX_STDIN_CHARS // 2
        omitted = len(text) - 2 * half
        io.err.write(
            f"agentforge: stdin has {len(text):,} characters; sending the first and last "
            f"{half:,} (skipping {omitted:,} in the middle)\n"
        )
        text = f"{text[:half]}\n[... {omitted:,} characters omitted ...]\n{text[-half:]}"
    block = f"<stdin>\n{text.rstrip()}\n</stdin>"
    if prompt:
        return f"{block}\n\n{prompt}"
    return (
        f"{block}\n\nThis was piped in on stdin without a question. Explain what it is and "
        "point out anything that needs attention."
    )


def cmd_ask(args: argparse.Namespace, io: _Streams, model: ModelBackend | None) -> int:
    prompt = " ".join(args.prompt).strip()
    if not isatty(io.stdin):
        piped = io.stdin.read()
        if piped.strip():
            prompt = attach_stdin(prompt, piped, io)
    if not prompt:
        raise UsageError('nothing to ask; try: agentforge ask "why is my disk full?"')
    renderer = make_renderer(args, io, json_mode=args.json)
    agent = build_agent(args, renderer, model)
    session = agent.session()
    try:
        result = run_request(session, prompt, renderer, agent.model.model)
    finally:
        if args.save:
            session.save(args.save)
    if result is None:
        return 1
    renderer.finish(result)
    if args.json:
        io.out.write(json.dumps(result_json(result), indent=2, ensure_ascii=False) + "\n")
    return exit_code(result)


def result_json(result: RunResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "stop_reason": result.stop_reason,
        "model": result.model,
        "turns": result.turns,
        "usage": result.usage.to_dict(),
        "tool_calls": [
            {
                "name": call.name,
                "input": call.input,
                "is_error": call.result.is_error if call.result else None,
                "result": call.result.content if call.result else None,
            }
            for call in result.tool_calls
        ],
    }


# ----------------------------------------------------------------------------- chat

CHAT_HELP = """\
Commands:
  /help            show this help
  /tools           list the tools available in this session
  /usage           token usage and estimated cost so far
  /save [FILE]     save the conversation as JSON
  /clear           start a new conversation
  /exit            quit (or Ctrl-D)
End a line with \\ to continue on the next line. Ctrl-C interrupts the current request."""


def cmd_chat(args: argparse.Namespace, io: _Streams, model: ModelBackend | None) -> int:
    if not (isatty(io.stdin) and isatty(io.out)):
        raise UsageError(
            "chat needs an interactive terminal; in scripts and pipes use `agentforge ask`"
        )
    renderer = make_renderer(args, io)
    agent = build_agent(args, renderer, model)
    session = agent.session()
    history = setup_readline()
    style = renderer.style
    mode = "auto (no approvals)" if args.yes else args.mode
    model_name = agent.model.model
    effort = getattr(agent.model, "effort", None)
    io.out.write(
        f"{style.bold('AgentForge')} {__version__} · {model_name}"
        + (f" (effort {effort})" if effort else "")
        + f" · {len(session.tools)} tools · approvals: {mode}\n"
        + style.dim(f"workspace: {agent.workspace.root}\n")
        + style.dim("Type a request, /help for commands, Ctrl-D to quit.\n")
    )
    prompt = "\001\033[1;36m\002❯\001\033[0m\002 " if style.enabled else "> "
    try:
        while True:
            try:
                text = read_message(prompt)
            except EOFError:
                io.out.write("\n")
                break
            except KeyboardInterrupt:
                io.out.write("^C\n")
                continue
            if not text.strip():
                continue
            if text.startswith("/"):
                next_session = slash_command(text.strip(), session, agent, renderer, io)
                if next_session is None:
                    break
                session = next_session
                continue
            result = run_request(session, text, renderer, model_name)
            if result is not None:
                renderer.finish(result)
            if args.save:
                session.save(args.save)
    finally:
        if history is not None:
            history()
    if session.usage.output_tokens:
        io.out.write(style.dim(f"Session: {usage_summary(session.usage, model_name)}\n"))
    return 0


def read_message(prompt: str) -> str:
    lines = [input(prompt)]
    while lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]
        lines.append(input("… "))
    return "\n".join(lines)


def slash_command(
    command: str, session: Session, agent: Agent, renderer: Renderer, io: _Streams
) -> Session | None:
    """Handle a /command. Returns the session to continue with, or None to quit."""
    name, _, argument = command.partition(" ")
    name = name.lower()
    if name in ("/exit", "/quit", "/q"):
        return None
    if name in ("/help", "/?"):
        io.out.write(CHAT_HELP + "\n")
    elif name == "/tools":
        for tool in session.tools.values():
            io.out.write(
                f"  {tool.name:<17} {tool.risk.value:<8} {first_sentence(tool.description)}\n"
            )
    elif name == "/usage":
        io.out.write(f"{usage_summary(session.usage, agent.model.model)}\n")
    elif name == "/save":
        path = argument.strip() or f"agentforge-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
        saved = session.save(path)
        io.out.write(f"Saved the conversation to {saved}\n")
    elif name in ("/clear", "/new"):
        io.out.write("Started a new conversation.\n")
        return agent.session()
    else:
        io.out.write(f"Unknown command {name}; try /help\n")
    return session


def setup_readline() -> Any:
    """Enable line editing and persistent history; returns a function that saves it."""
    try:
        import readline
    except ImportError:  # Windows without pyreadline
        return None
    state_home = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    path = Path(state_home) / "agentforge" / "history"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            readline.read_history_file(path)
    except OSError:
        pass
    readline.set_history_length(1000)

    def save() -> None:
        try:
            readline.write_history_file(path)
        except OSError:
            pass

    return save


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
