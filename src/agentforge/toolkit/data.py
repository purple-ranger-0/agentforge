"""Precise data utilities: the conversions models shouldn't do in their heads.

JWTs, base64/hex/URL encodings, hashes, timestamps and time zones, cron schedules,
regexes, arithmetic, subnets, secrets, and JSON queries.
"""

from __future__ import annotations

import ast
import base64
import binascii
import datetime as dt
import email.utils
import hashlib
import html
import ipaddress
import itertools
import json
import math
import operator
import re
import secrets
import string
import subprocess
import sys
import urllib.parse
import uuid
import zlib
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..errors import ToolError
from ..tools import ToolContext, tool
from ._util import clamp, human_size, relative_time

_UTC = dt.timezone.utc

# ----------------------------------------------------------------------------- JWT


def _b64url(segment: str) -> bytes:
    segment = segment.strip()
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


@tool
def decode_jwt(token: str) -> dict[str, Any]:
    """Decode a JWT and show its header and claims, with exp/iat/nbf as readable UTC times
    and whether it has expired. The signature is NOT verified. Use it to debug auth
    failures ("why is my token rejected?") or check which scopes a token carries.

    Args:
        token: The JWT (a leading "Bearer " is ignored).
    """
    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    parts = token.split(".")
    if len(parts) == 5:
        raise ToolError(
            "this is an encrypted JWE (5 parts); its claims can't be read without the key"
        )
    if len(parts) != 3:
        raise ToolError(f"a JWT has 3 dot-separated parts; this has {len(parts)}")
    try:
        header = json.loads(_b64url(parts[0]))
        payload = json.loads(_b64url(parts[1]))
    except (ValueError, binascii.Error) as exc:
        raise ToolError(f"not a valid JWT: {exc}") from exc
    now = dt.datetime.now(_UTC)
    times: dict[str, str] = {}
    moments: dict[str, dt.datetime] = {}
    if isinstance(payload, dict):
        for claim in ("exp", "nbf", "iat", "auth_time"):
            value = payload.get(claim)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    when = dt.datetime.fromtimestamp(value, _UTC)
                except (OverflowError, OSError, ValueError):
                    continue
                moments[claim] = when
                times[claim] = f"{when.isoformat()} ({relative_time(when, now)})"
    if "nbf" in moments and moments["nbf"] > now:
        status = "not valid yet (nbf is in the future)"
    elif "exp" in moments:
        status = "expired" if moments["exp"] <= now else "not expired"
    else:
        status = "no exp claim, so it never expires"
    return {
        "header": header,
        "payload": payload,
        "times_utc": times,
        "status": status,
        "signature": "present but NOT verified" if parts[2] else "none (unsigned token)",
    }


# ------------------------------------------------------------------------ encodings


def _bytes_to_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        shown = raw[:1024].hex()
        more = f"... ({len(raw)} bytes total)" if len(raw) > 1024 else ""
        return f"(not UTF-8 text: {len(raw)} bytes) hex: {shown}{more}"


@tool
def convert_encoding(
    text: str,
    codec: Literal["base64", "base64url", "hex", "url", "html"],
    direction: Literal["encode", "decode"] = "decode",
) -> str:
    """Encode or decode text exactly: base64, base64url, hex, URL percent-encoding, or HTML
    entities. Decoded bytes that aren't UTF-8 are shown as hex. URL decoding turns "+" into
    a space, as in query strings.

    Args:
        text: The input text.
        codec: The encoding.
        direction: "encode" or "decode".
    """
    try:
        if direction == "encode":
            data = text.encode("utf-8")
            if codec == "base64":
                return base64.b64encode(data).decode("ascii")
            if codec == "base64url":
                return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
            if codec == "hex":
                return data.hex()
            if codec == "url":
                return urllib.parse.quote(text, safe="")
            return html.escape(text)
        if codec in ("base64", "base64url"):
            cleaned = "".join(text.split())
            padded = cleaned + "=" * (-len(cleaned) % 4)
            if codec == "base64url" or "-" in cleaned or "_" in cleaned:
                return _bytes_to_text(base64.urlsafe_b64decode(padded))
            return _bytes_to_text(base64.b64decode(padded, validate=True))
        if codec == "hex":
            tokens = re.split(r"[\s:,]+", text.strip())
            cleaned = "".join(t[2:] if t.lower().startswith("0x") else t for t in tokens)
            return _bytes_to_text(bytes.fromhex(cleaned))
        if codec == "url":
            return urllib.parse.unquote_plus(text)
        return html.unescape(text)
    except (ValueError, binascii.Error) as exc:
        raise ToolError(f"couldn't {direction} as {codec}: {exc}") from exc


@tool
def compute_hash(
    ctx: ToolContext,
    text: str | None = None,
    path: str | None = None,
    algorithm: Literal[
        "sha256", "sha1", "md5", "sha512", "sha3_256", "blake2b", "crc32"
    ] = "sha256",
) -> str:
    """Compute a checksum of a text string or of a file (pass exactly one). Use it to
    verify downloads, compare files, or produce hashes for configs and caches.

    Args:
        text: Text to hash (UTF-8 encoded).
        path: File to hash, relative to the workspace root.
        algorithm: Hash algorithm.
    """
    if (text is None) == (path is None):
        raise ToolError("pass exactly one of text or path")
    if algorithm == "crc32":
        crc = 0

        def update(chunk: bytes) -> None:
            nonlocal crc
            crc = zlib.crc32(chunk, crc)

        def digest() -> str:
            return f"{crc:08x}"

    else:
        hasher = hashlib.new(algorithm)
        update = hasher.update

        def digest() -> str:
            return hasher.hexdigest()

    if text is not None:
        data = text.encode("utf-8")
        update(data)
        source = f"{len(data)} bytes of UTF-8 text"
    else:
        file = ctx.workspace.resolve(path or "", kind="file")
        size = 0
        with file.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                update(chunk)
                size += len(chunk)
        source = f"{ctx.workspace.display(file)}, {human_size(size)}"
    return f"{algorithm}: {digest()}  ({source})"


# --------------------------------------------------------------------------- time


def _zone(name: str) -> dt.tzinfo:
    if name.strip().upper() in ("UTC", "Z", "GMT"):
        return _UTC
    try:
        return ZoneInfo(name.strip())
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolError(
            f"unknown time zone {name!r}; use an IANA name like 'Europe/Berlin' "
            "or 'America/New_York'"
        ) from exc


def _parse_time(value: str, naive_zone: dt.tzinfo, now: dt.datetime) -> tuple[dt.datetime, str]:
    text = value.strip()
    if text.lower() == "now":
        return now, "the current time"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        number = float(text)
        magnitude = abs(number)
        if magnitude < 1e11:
            seconds, unit = number, "Unix seconds"
        elif magnitude < 1e14:
            seconds, unit = number / 1e3, "Unix milliseconds"
        elif magnitude < 1e17:
            seconds, unit = number / 1e6, "Unix microseconds"
        else:
            seconds, unit = number / 1e9, "Unix nanoseconds"
        try:
            return dt.datetime.fromtimestamp(seconds, _UTC), unit
        except (OverflowError, OSError, ValueError) as exc:
            raise ToolError(f"timestamp {value!r} is out of range") from exc
    iso = text[:-1] + "+00:00" if text[-1:] in ("Z", "z") else text
    parsed: dt.datetime | None
    try:
        parsed = dt.datetime.fromisoformat(iso)
        kind = "ISO 8601"
    except ValueError:
        parsed = None
    if parsed is None:
        try:
            parsed = email.utils.parsedate_to_datetime(text)
            kind = "RFC 2822 / HTTP date"
        except (TypeError, ValueError, IndexError):
            parsed = None
    if parsed is None:
        raise ToolError(
            f"couldn't parse {value!r}; use Unix epoch seconds or milliseconds, ISO 8601 "
            "(2026-01-31T12:00:00Z) or an RFC 2822 date (Sat, 31 Jan 2026 12:00:00 GMT)"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=naive_zone)
        kind += f" (no offset given; read as {naive_zone})"
    return parsed, kind


@tool
def convert_time(value: str, timezone: str = "UTC", input_timezone: str = "UTC") -> dict[str, Any]:
    """Convert a timestamp between Unix epoch (s/ms/µs/ns, detected by size), ISO 8601,
    RFC 2822 and any IANA time zone, and say how long ago or ahead it is. Use it for exact
    date arithmetic on timestamps from logs, tokens, databases or APIs. Pass "now" for the
    current time.

    Args:
        value: The timestamp, e.g. "1767225600", "1767225600000",
            "2026-01-01T09:30:00+01:00" or "now".
        timezone: IANA zone to show the result in, e.g. "America/New_York".
        input_timezone: Zone for inputs without an offset, e.g. "2026-03-01 09:00".
    """
    zone = _zone(timezone)
    now = dt.datetime.now(_UTC)
    when, kind = _parse_time(value, _zone(input_timezone), now)
    local = when.astimezone(zone)
    return {
        "interpreted_as": kind,
        "utc": when.astimezone(_UTC).isoformat(),
        "in_timezone": f"{local.isoformat()} ({timezone}, {local.tzname()})",
        "weekday": local.strftime("%A"),
        "unix_seconds": int(when.timestamp()),
        "unix_milliseconds": int(when.timestamp() * 1000),
        "relative_to_now": relative_time(when, now),
    }


# --------------------------------------------------------------------------- cron

_CRON_FIELDS = (
    ("minute", 0, 59, None),
    ("hour", 0, 23, None),
    ("day_of_month", 1, 31, None),
    (
        "month",
        1,
        12,
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    ),
    ("day_of_week", 0, 7, ("sun", "mon", "tue", "wed", "thu", "fri", "sat")),
)
_CRON_MACROS = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}
_DAY_NAMES = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _cron_value(token: str, field: str, names: tuple[str, ...] | None, low: int) -> int:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    if names and token[:3] in names and len(token) >= 3:
        return names.index(token[:3]) + (1 if low == 1 else 0)
    raise ToolError(f"invalid {field} value {token!r}")


def _cron_field(
    text: str, field: str, low: int, high: int, names: tuple[str, ...] | None
) -> set[int]:
    values: set[int] = set()
    for item in text.split(","):
        base, slash, step_text = item.partition("/")
        step = 1
        if slash:
            if not step_text.isdigit() or int(step_text) == 0:
                raise ToolError(f"invalid step {step_text!r} in the {field} field")
            step = int(step_text)
        if base in ("*", "?"):
            start, end = low, high
        elif "-" in base:
            first, _, last = base.partition("-")
            start, end = _cron_value(first, field, names, low), _cron_value(last, field, names, low)
        elif base:
            start = _cron_value(base, field, names, low)
            end = high if slash else start
        else:
            raise ToolError(f"empty item in the {field} field")
        if start > end:
            raise ToolError(f"range {base!r} in the {field} field runs backwards")
        if start < low or end > high:
            raise ToolError(f"{field} values must be between {low} and {high}")
        values.update(range(start, end + 1, step))
    return values


def _compress(values: list[int], labels: tuple[str, ...] | None = None, offset: int = 0) -> str:
    def label(v: int) -> str:
        return labels[v - offset] if labels else str(v)

    parts = []
    for _, group in itertools.groupby(enumerate(values), key=lambda iv: iv[1] - iv[0]):
        run = [v for _, v in group]
        if len(run) >= 3:
            parts.append(f"{label(run[0])}-{label(run[-1])}")
        else:
            parts.extend(label(v) for v in run)
    return ",".join(parts)


@tool
def cron_schedule(
    expression: str, count: int = 5, timezone: str = "UTC", after: str | None = None
) -> dict[str, Any]:
    """Expand a cron expression and list its next run times. Standard 5 fields (minute,
    hour, day of month, month, day of week) with *, lists, ranges, steps and names, or
    @hourly/@daily/@weekly/@monthly/@yearly. Use it to explain or sanity-check crontab,
    Kubernetes CronJob and CI schedules.

    Args:
        expression: The cron expression, e.g. "*/15 9-17 * * mon-fri".
        count: How many upcoming runs to list (1-50).
        timezone: IANA zone the schedule runs in.
        after: List runs after this time (any format convert_time accepts). Default: now.
    """
    zone = _zone(timezone)
    expr = " ".join(expression.split())
    fields_text = _CRON_MACROS.get(expr.lower(), expr).split()
    if len(fields_text) != 5:
        raise ToolError(
            "expected 5 fields (minute hour day-of-month month day-of-week), "
            f"got {len(fields_text)}"
        )
    parsed = [
        _cron_field(text, name, low, high, names)
        for text, (name, low, high, names) in zip(fields_text, _CRON_FIELDS, strict=True)
    ]
    minutes, hours, days, months, weekdays = parsed
    weekdays = {0 if d == 7 else d for d in weekdays}
    # Vixie cron: when both day fields are restricted, a day matches if EITHER does.
    dom_restricted = not fields_text[2].startswith("*")
    dow_restricted = not fields_text[4].startswith("*")

    def day_matches(day: dt.date) -> bool:
        if day.month not in months:
            return False
        dom_ok, dow_ok = day.day in days, day.isoweekday() % 7 in weekdays
        if dom_restricted and dow_restricted:
            return dom_ok or dow_ok
        return (dom_ok or not dom_restricted) and (dow_ok or not dow_restricted)

    now = dt.datetime.now(_UTC)
    start = _parse_time(after, zone, now)[0] if after else now
    start = start.astimezone(zone).replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    count = clamp(count, 1, 50)
    runs: list[str] = []
    day = start.date()
    for _ in range(366 * 8):  # far enough for Feb 29 schedules
        if day_matches(day):
            for hour in sorted(hours):
                for minute in sorted(minutes):
                    candidate = dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
                    if candidate >= start:
                        runs.append(f"{candidate.isoformat()} ({candidate:%a})")
                    if len(runs) == count:
                        break
                if len(runs) == count:
                    break
        if len(runs) == count:
            break
        day += dt.timedelta(days=1)

    result: dict[str, Any] = {
        "expression": expression,
        "timezone": timezone,
        "fields": {
            "minute": _compress(sorted(minutes)),
            "hour": _compress(sorted(hours)),
            "day_of_month": "*" if not dom_restricted else _compress(sorted(days)),
            "month": "*"
            if months == set(range(1, 13))
            else _compress(sorted(months), _MONTH_NAMES, 1),
            "day_of_week": "*" if not dow_restricted else _compress(sorted(weekdays), _DAY_NAMES),
        },
        "next_runs": runs,
    }
    if dom_restricted and dow_restricted:
        result["note"] = "both day fields are set, so it runs when EITHER matches"
    if not runs:
        result["note"] = "this schedule never fires (e.g. a day that doesn't exist in that month)"
    return result


# -------------------------------------------------------------------------- regex

_REGEX_TIMEOUT = 5.0  # seconds
_REGEX_FLAGS = {
    "IGNORECASE": re.I,
    "MULTILINE": re.M,
    "DOTALL": re.S,
    "VERBOSE": re.X,
    "ASCII": re.A,
}

# Runs in a separate interpreter so a catastrophically backtracking pattern can be killed.
_REGEX_WORKER = r"""
import json, re, sys
job = json.load(sys.stdin)
try:
    rx = re.compile(job["pattern"], job["flags"])
except re.error as exc:
    print(json.dumps({"error": "invalid regular expression: %s" % exc}))
    raise SystemExit
found, total = [], 0
for m in rx.finditer(job["text"]):
    total += 1
    if len(found) < job["max"]:
        found.append({"match": m.group(0), "span": [m.start(), m.end()],
                      "groups": list(m.groups()), "named_groups": m.groupdict()})
    if total >= 10000:
        break
print(json.dumps({"full_match": rx.fullmatch(job["text"]) is not None, "match_count": total,
                  "matches": found, "group_count": rx.groups}))
"""


@tool
def regex_test(
    pattern: str,
    text: str,
    flags: list[Literal["IGNORECASE", "MULTILINE", "DOTALL", "VERBOSE", "ASCII"]] | None = None,
    max_matches: int = 50,
) -> dict[str, Any]:
    """Test a regular expression (Python `re` syntax) against sample text: every match with
    its position and capture groups, and whether the whole text matches. Use it to build
    or debug a regex before it goes into code, configs or log queries. Patterns that take
    longer than 5 seconds (catastrophic backtracking) are stopped.

    Args:
        pattern: The regular expression.
        text: Text to run it against (up to 200,000 characters).
        flags: Regex flags to enable.
        max_matches: Maximum matches to list (1-500).
    """
    if len(text) > 200_000:
        raise ToolError("text is too long (max 200,000 characters)")
    flag_value = 0
    for name in flags or []:
        flag_value |= _REGEX_FLAGS[name]
    job = json.dumps(
        {"pattern": pattern, "text": text, "flags": flag_value, "max": clamp(max_matches, 1, 500)}
    )
    if not sys.executable:
        raise ToolError("can't start a Python subprocess to test the regex safely")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _REGEX_WORKER],
            input=job,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_REGEX_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(
            f"the regex ran for over {_REGEX_TIMEOUT:g} seconds on this text, which usually "
            "means catastrophic "
            "backtracking (nested quantifiers like (a+)+ or (.*)*); simplify the pattern"
        ) from exc
    if proc.returncode != 0:
        raise ToolError(f"regex worker failed: {proc.stderr.strip()[-500:]}")
    result = json.loads(proc.stdout)
    if "error" in result:
        raise ToolError(result["error"])
    return {"pattern": pattern, "flags": flags or [], **result}


# --------------------------------------------------------------------- calculator

_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}
_COMPARE_OPS: dict[type[ast.cmpop], Any] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def _factorial(n: int) -> int:
    if not isinstance(n, int) or n < 0 or n > 5000:
        raise ToolError("factorial needs an integer between 0 and 5000")
    return math.factorial(n)


_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    "factorial": _factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "comb": math.comb,
    "perm": math.perm,
    "int": int,
    "float": float,
}


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        _check_size(node.op, left, right)
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if type(op) not in _COMPARE_OPS:
                raise ToolError("unsupported comparison")
            right = _evaluate(comparator)
            if not _COMPARE_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate(element) for element in node.elts]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ToolError(f"unsupported expression: {ast.unparse(node)!r}")


def _check_size(op: ast.operator, left: Any, right: Any) -> None:
    if not (isinstance(left, int) and isinstance(right, int)):
        return
    if (
        isinstance(op, ast.Pow)
        and right > 0
        and abs(left) > 1
        and right * math.log2(abs(left)) > 1_000_000
    ):
        raise ToolError("result is too large (over a million bits)")
    if isinstance(op, ast.LShift) and right > 1_000_000:
        raise ToolError("shift is too large")
    if isinstance(op, ast.Mult) and left.bit_length() + right.bit_length() > 2_000_000:
        raise ToolError("result is too large")


def _format_number(value: Any, show_hex: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        if value.bit_length() > 13_000:  # beyond what str() will print (~4300 digits)
            exponent = math.floor(math.log10(abs(value)))
            mantissa = 10 ** (math.log10(abs(value)) - exponent)
            return (
                f"≈ {'-' if value < 0 else ''}{mantissa:.6f}e{exponent} ({exponent + 1:,} digits)"
            )
        text = str(value)
        extras = []
        if abs(value) >= 10_000:
            extras.append(f"{value:,}")
        if show_hex and 0 <= value < 2**64:
            extras.append(hex(value))
        return f"{text} ({'; '.join(extras)})" if extras else text
    if math.isfinite(value) and value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return f"{value:.15g}"


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. "5 * 1024**3", "0xff & 0b1010",
    "(1 - 0.995) * 30 * 24 * 60", "log2(65536)", "comb(52, 5)". Supports + - * / // % **,
    bitwise & | ^ ~ << >>, comparisons, hex/octal/binary literals, the constants pi, e and
    tau, and abs round min max sum sqrt exp log log2 log10 sin cos tan asin acos atan atan2
    hypot degrees radians ceil floor trunc factorial gcd lcm comb perm int float. Use it
    instead of mental arithmetic: capacity planning, unit conversions, SLO budgets, masks.

    Args:
        expression: The expression to evaluate.
    """
    if len(expression) > 2000:
        raise ToolError("expression is too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc
    try:
        value = _evaluate(tree)
    except ToolError:
        raise
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise ToolError(f"couldn't evaluate: {exc}") from exc
    bitwise = re.search(r"0[xXbBoO]|[&|^~]|<<|>>", expression) is not None
    show_hex = bitwise or (isinstance(value, int) and value >= 2**16)
    return f"{expression.strip()} = {_format_number(value, show_hex)}"


# ------------------------------------------------------------------------- subnets


@tool
def cidr_info(
    network: str, check: list[str] | None = None, split_prefix: int | None = None
) -> dict[str, Any]:
    """Describe an IPv4 or IPv6 network in CIDR notation: network and broadcast address,
    netmask, usable host range and count, and address type (private, public, ...).
    Optionally test which addresses or subnets fall inside it, or split it into smaller
    subnets. Use it for subnetting, firewall rules, VPC/Kubernetes network plans and
    "is this IP in that range?".

    Args:
        network: Network like "10.0.0.0/22" or "2001:db8::/48" (a bare IP means a single address).
        check: Addresses or subnets to test for membership, e.g. ["10.0.3.17", "10.0.4.0/24"].
        split_prefix: Split into subnets of this prefix length (lists up to 64).
    """
    try:
        given = ipaddress.ip_interface(network.strip())
    except ValueError as exc:
        raise ToolError(f"invalid network: {exc}") from exc
    net = given.network
    info: dict[str, Any] = {"network": str(net), "version": f"IPv{net.version}"}
    if given.ip != net.network_address:
        info["note"] = f"{network!r} has host bits set; the network it belongs to is {net}"
    info["prefix_length"] = net.prefixlen
    info["netmask"] = str(net.netmask)
    info["hostmask"] = str(net.hostmask)
    info["total_addresses"] = net.num_addresses
    if net.version == 4:
        info["broadcast"] = str(net.broadcast_address)
        if net.prefixlen >= 31:
            usable_first, usable_last, usable = net[0], net[-1], net.num_addresses
        else:
            usable_first, usable_last, usable = net[1], net[-2], net.num_addresses - 2
    else:
        usable_first, usable_last, usable = net[0], net[-1], net.num_addresses
    info["usable_hosts"] = usable
    info["host_range"] = f"{usable_first} - {usable_last}"
    kinds = [
        name
        for name in (
            "private",
            "global",
            "loopback",
            "link_local",
            "multicast",
            "reserved",
            "unspecified",
        )
        if getattr(net, f"is_{name}", False)
    ]
    info["type"] = kinds or ["unclassified"]
    if check:
        membership = {}
        for item in check:
            try:
                candidate = ipaddress.ip_network(item.strip(), strict=False)
            except ValueError:
                membership[item] = "invalid address"
                continue
            if candidate.version != net.version:
                membership[item] = f"different IP version (IPv{candidate.version})"
            else:
                membership[item] = "inside" if candidate.subnet_of(net) else "outside"
        info["membership"] = membership
    if split_prefix is not None:
        if not net.prefixlen <= split_prefix <= net.max_prefixlen:
            raise ToolError(f"split_prefix must be between {net.prefixlen} and {net.max_prefixlen}")
        total = 2 ** (split_prefix - net.prefixlen)
        info["subnets"] = {
            "count": total,
            "listed": [str(s) for s in itertools.islice(net.subnets(new_prefix=split_prefix), 64)],
        }
    return info


# ------------------------------------------------------------------------ secrets

_PASSWORD_SYMBOLS = "!@#$%^&*-_=+?"


@tool
def generate_secret(
    kind: Literal["urlsafe", "hex", "password", "uuid4"] = "urlsafe",
    length: int = 32,
    count: int = 1,
) -> str:
    """Generate cryptographically secure random values (Python's `secrets` module):
    URL-safe or hex tokens for API keys, session or webhook secrets; passwords with
    upper/lower case, digits and symbols; or UUIDv4s.

    Args:
        kind: What to generate.
        length: Bytes of randomness for tokens; characters for passwords (8-256). Ignored for UUIDs.
        count: How many values (1-50).
    """
    length, count = clamp(length, 8, 256), clamp(count, 1, 50)
    alphabet = string.ascii_letters + string.digits + _PASSWORD_SYMBOLS
    values = []
    for _ in range(count):
        if kind == "urlsafe":
            values.append(secrets.token_urlsafe(length))
        elif kind == "hex":
            values.append(secrets.token_hex(length))
        elif kind == "uuid4":
            values.append(str(uuid.uuid4()))
        else:
            while True:
                candidate = "".join(secrets.choice(alphabet) for _ in range(length))
                if (
                    any(c.islower() for c in candidate)
                    and any(c.isupper() for c in candidate)
                    and any(c.isdigit() for c in candidate)
                    and any(c in _PASSWORD_SYMBOLS for c in candidate)
                ):
                    values.append(candidate)
                    break
    return "\n".join(values)


# ---------------------------------------------------------------------------- JSON

_MAX_JSON_BYTES = 50 * 1024 * 1024
_WILDCARD = object()


def _parse_query(query: str) -> list[Any]:
    tokens: list[Any] = []
    i, text = 0, query.strip()
    if text.startswith("$"):
        text = text[1:]
    while i < len(text):
        char = text[i]
        if char == ".":
            i += 1
        elif char == "[":
            end = text.find("]", i)
            if end == -1:
                raise ToolError(f"unclosed '[' in query {query!r}")
            inner = text[i + 1 : end].strip()
            if inner == "*":
                tokens.append(_WILDCARD)
            elif re.fullmatch(r"-?\d+", inner):
                tokens.append(int(inner))
            elif len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in "'\"":
                tokens.append(inner[1:-1])
            else:
                raise ToolError(f"invalid index [{inner}] in query {query!r}")
            i = end + 1
        else:
            end = i
            while end < len(text) and text[end] not in ".[":
                end += 1
            name = text[i:end]
            tokens.append(_WILDCARD if name == "*" else name)
            i = end
    return tokens


def _select(data: Any, tokens: list[Any]) -> tuple[list[Any], bool]:
    values, wildcard = [data], False
    for depth, token in enumerate(tokens):
        selected = []
        for value in values:
            if token is _WILDCARD:
                wildcard = True
                if isinstance(value, dict):
                    selected.extend(value.values())
                elif isinstance(value, list):
                    selected.extend(value)
            elif isinstance(token, int):
                if isinstance(value, list) and -len(value) <= token < len(value):
                    selected.append(value[token])
                elif not wildcard:
                    raise ToolError(f"index [{token}] at step {depth + 1}: {_describe(value)}")
            elif isinstance(value, dict) and token in value:
                selected.append(value[token])
            elif not wildcard:
                keys = ", ".join(list(value)[:30]) if isinstance(value, dict) else ""
                hint = (
                    f"; available keys: {keys}"
                    if keys
                    else f"; the value there is {_describe(value)}"
                )
                raise ToolError(f"key {token!r} not found{hint}")
        values = selected
    return values, wildcard


def _describe(value: Any) -> str:
    if isinstance(value, dict):
        return f"an object with {len(value)} keys"
    if isinstance(value, list):
        return f"an array of {len(value)} items"
    return f"a {type(value).__name__}"


def _outline(value: Any, depth: int = 0, max_depth: int = 3) -> list[str]:
    pad = "  " * depth
    if isinstance(value, dict):
        lines = []
        for key, child in itertools.islice(value.items(), 40):
            kind = _describe(child) if isinstance(child, (dict, list)) else type(child).__name__
            lines.append(f"{pad}{key}: {kind}")
            if isinstance(child, (dict, list)) and depth + 1 < max_depth:
                lines.extend(_outline(child, depth + 1, max_depth))
        if len(value) > 40:
            lines.append(f"{pad}... {len(value) - 40} more keys")
        return lines
    if isinstance(value, list) and value:
        return [f"{pad}[0]: {_describe(value[0])}", *_outline(value[0], depth + 1, max_depth)]
    return []


@tool
def json_query(
    ctx: ToolContext, query: str = "", path: str | None = None, text: str | None = None
) -> str:
    """Pull data out of a JSON or JSON Lines document with a path query, without reading the
    whole document into the conversation. Use it on big files like package-lock.json,
    terraform state, API dumps or kubectl -o json output.

    Query syntax: dot-separated keys and [index], with * as a wildcard, e.g. "dependencies",
    "items[0].metadata.name", "items[*].status.phase", "*.version". An empty query shows the
    whole document, or an outline of its structure when it is large.

    Args:
        query: The path to select.
        path: JSON file in the workspace (pass this or text).
        text: JSON text (pass this or path).
    """
    if (path is None) == (text is None):
        raise ToolError("pass exactly one of path or text")
    if path is not None:
        file = ctx.workspace.resolve(path, kind="file")
        if file.stat().st_size > _MAX_JSON_BYTES:
            raise ToolError(f"{path} is larger than {human_size(_MAX_JSON_BYTES)}")
        raw = file.read_text(encoding="utf-8", errors="replace")
    else:
        raw = text or ""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        lines = [line for line in raw.splitlines() if line.strip()]
        try:
            data = [json.loads(line) for line in lines]  # JSON Lines
        except ValueError:
            raise ToolError(f"not valid JSON: {exc}") from exc
    values, wildcard = _select(data, _parse_query(query))
    result = values if wildcard else values[0]
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if len(rendered) <= 30_000:
        return rendered
    outline = "\n".join(_outline(result)) or _describe(result)
    return (
        f"The result is large ({len(rendered):,} characters of JSON), so here's its structure "
        f"instead. Query a narrower path to see values.\n{_describe(result)}\n{outline}"
    )


TOOLS = (
    decode_jwt,
    convert_encoding,
    compute_hash,
    convert_time,
    cron_schedule,
    regex_test,
    calculate,
    cidr_info,
    generate_secret,
    json_query,
)
