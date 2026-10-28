from __future__ import annotations

import base64
import datetime as dt
import json
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agentforge.errors import ToolError
from agentforge.toolkit import data

Run = Callable[..., Any]


def make_jwt(payload: dict[str, Any], header: dict[str, Any] | None = None) -> str:
    def enc(obj: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{enc(header or {'alg': 'HS256', 'typ': 'JWT'})}.{enc(payload)}.c2ln"


def test_decode_jwt(run: Run) -> None:
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    token = make_jwt({"sub": "42", "scope": "read", "exp": now - 3600, "iat": now - 7200})
    result = run(data.decode_jwt, token=f"Bearer {token}")
    assert result["header"] == {"alg": "HS256", "typ": "JWT"}
    assert result["payload"]["scope"] == "read"
    assert result["status"] == "expired"
    assert result["times_utc"]["exp"].endswith("(1 hour ago)")
    assert result["signature"] == "present but NOT verified"
    fresh = run(data.decode_jwt, token=make_jwt({"exp": now + 86400 * 2 + 60}))
    assert fresh["status"] == "not expired" and "(in 2 days" in fresh["times_utc"]["exp"]
    assert run(data.decode_jwt, token=make_jwt({"sub": "x"}))["status"].startswith("no exp claim")
    assert run(data.decode_jwt, token=make_jwt({"nbf": now + 999}))["status"].startswith(
        "not valid yet"
    )


@pytest.mark.parametrize(
    ("token", "message"),
    [("a.b", "has 2"), ("a.b.c.d.e", "encrypted JWE"), ("!!!.???.x", "not a valid JWT")],
)
def test_decode_jwt_errors(run: Run, token: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        run(data.decode_jwt, token=token)


@pytest.mark.parametrize(
    ("text", "codec", "encoded"),
    [
        ("hello world", "base64", "aGVsbG8gd29ybGQ="),
        ("hi?>", "base64url", "aGk_Pg"),
        ("AB", "hex", "4142"),
        ("a b&c/d", "url", "a%20b%26c%2Fd"),
        ("<a & b>", "html", "&lt;a &amp; b&gt;"),
    ],
)
def test_convert_encoding_round_trips(run: Run, text: str, codec: str, encoded: str) -> None:
    assert run(data.convert_encoding, text=text, codec=codec, direction="encode") == encoded
    assert run(data.convert_encoding, text=encoded, codec=codec) == text


def test_convert_encoding_lenient_inputs(run: Run) -> None:
    assert run(data.convert_encoding, text="aGVsbG8", codec="base64") == "hello"  # missing padding
    assert run(data.convert_encoding, text="0x41 0x42", codec="hex") == "AB"
    assert run(data.convert_encoding, text="41:42", codec="hex") == "AB"
    assert run(data.convert_encoding, text="a+b%21", codec="url") == "a b!"
    assert run(data.convert_encoding, text="//8=", codec="base64").startswith(
        "(not UTF-8 text: 2 bytes) hex: ffff"
    )
    with pytest.raises(ToolError, match="couldn't decode as hex"):
        run(data.convert_encoding, text="zz", codec="hex")


def test_compute_hash(run: Run, tmp_path: Path) -> None:
    assert run(data.compute_hash, text="hello").startswith(
        "sha256: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert run(data.compute_hash, text="hello", algorithm="md5").startswith(
        "md5: 5d41402abc4b2a76b9719d911017c592"
    )
    assert run(data.compute_hash, text="hello", algorithm="crc32").startswith("crc32: 3610a686")
    (tmp_path / "f.bin").write_bytes(b"hello")
    assert run(data.compute_hash, path="f.bin", algorithm="sha1") == (
        "sha1: aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d  (f.bin, 5 B)"
    )
    with pytest.raises(ToolError, match="exactly one"):
        run(data.compute_hash)
    with pytest.raises(ToolError, match="exactly one"):
        run(data.compute_hash, text="a", path="f.bin")


@pytest.mark.parametrize(
    ("value", "interpreted", "utc"),
    [
        ("1767225600", "Unix seconds", "2026-01-01T00:00:00+00:00"),
        ("1767225600000", "Unix milliseconds", "2026-01-01T00:00:00+00:00"),
        ("1767225600000000", "Unix microseconds", "2026-01-01T00:00:00+00:00"),
        ("1767225600000000000", "Unix nanoseconds", "2026-01-01T00:00:00+00:00"),
        ("2026-01-01T05:30:00+05:30", "ISO 8601", "2026-01-01T00:00:00+00:00"),
        ("2026-01-01T00:00:00Z", "ISO 8601", "2026-01-01T00:00:00+00:00"),
        ("Thu, 01 Jan 2026 00:00:00 GMT", "RFC 2822 / HTTP date", "2026-01-01T00:00:00+00:00"),
    ],
)
def test_convert_time_parses_formats(run: Run, value: str, interpreted: str, utc: str) -> None:
    result = run(data.convert_time, value=value)
    assert result["interpreted_as"] == interpreted
    assert result["utc"] == utc
    assert result["unix_seconds"] == 1767225600


def test_convert_time_zones(run: Run) -> None:
    result = run(
        data.convert_time,
        value="2026-07-01 09:00",
        input_timezone="America/New_York",
        timezone="Asia/Tokyo",
    )
    assert result["utc"] == "2026-07-01T13:00:00+00:00"
    assert result["in_timezone"] == "2026-07-01T22:00:00+09:00 (Asia/Tokyo, JST)"
    assert result["weekday"] == "Wednesday"
    assert "no offset given" in result["interpreted_as"]
    assert run(data.convert_time, value="now")["relative_to_now"] == "now"
    with pytest.raises(ToolError, match="unknown time zone"):
        run(data.convert_time, value="now", timezone="Mars/Olympus")
    with pytest.raises(ToolError, match="couldn't parse"):
        run(data.convert_time, value="next tuesday")


def test_cron_weekday_business_hours(run: Run) -> None:
    result = run(
        data.cron_schedule,
        expression="*/15 9-17 * * mon-fri",
        count=3,
        timezone="America/New_York",
        after="2026-10-30T17:40:00-04:00",
    )
    assert result["fields"] == {
        "minute": "0,15,30,45",
        "hour": "9-17",
        "day_of_month": "*",
        "month": "*",
        "day_of_week": "Mon-Fri",
    }
    assert result["next_runs"] == [
        "2026-10-30T17:45:00-04:00 (Fri)",
        "2026-11-02T09:00:00-05:00 (Mon)",  # skips the weekend and the DST change
        "2026-11-02T09:15:00-05:00 (Mon)",
    ]


def test_cron_macros_names_and_or_semantics(run: Run) -> None:
    yearly = run(data.cron_schedule, expression="@yearly", count=1, after="2026-03-01T00:00:00Z")
    assert yearly["next_runs"] == ["2027-01-01T00:00:00+00:00 (Fri)"]
    both = run(
        data.cron_schedule, expression="0 12 13 * fri", count=3, after="2026-03-01T00:00:00Z"
    )
    assert both["note"].startswith("both day fields are set")
    assert both["next_runs"][:2] == [
        "2026-03-06T12:00:00+00:00 (Fri)",
        "2026-03-13T12:00:00+00:00 (Fri)",
    ]
    leap = run(data.cron_schedule, expression="0 0 29 feb *", count=1, after="2026-03-01T00:00:00Z")
    assert leap["next_runs"] == ["2028-02-29T00:00:00+00:00 (Tue)"]
    never = run(data.cron_schedule, expression="0 0 31 2 *", count=1)
    assert never["next_runs"] == [] and "never fires" in never["note"]
    sunday7 = run(data.cron_schedule, expression="0 0 * * 7", count=1, after="2026-03-01T00:00:00Z")
    assert sunday7["next_runs"] == ["2026-03-08T00:00:00+00:00 (Sun)"]


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("* * * *", "expected 5 fields"),
        ("60 * * * *", "between 0 and 59"),
        ("*/0 * * * *", "invalid step"),
        ("5-1 * * * *", "runs backwards"),
        ("* * * foo *", "invalid month value"),
    ],
)
def test_cron_errors(run: Run, expression: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        run(data.cron_schedule, expression=expression)


def test_regex_test(run: Run) -> None:
    result = run(
        data.regex_test,
        pattern=r"(?P<user>\w+)@(?P<host>[\w.]+)",
        text="ada@example.com, bob@x.org",
    )
    assert result["match_count"] == 2
    assert result["matches"][0] == {
        "match": "ada@example.com",
        "span": [0, 15],
        "groups": ["ada", "example.com"],
        "named_groups": {"user": "ada", "host": "example.com"},
    }
    assert result["full_match"] is False
    assert (
        run(data.regex_test, pattern="^abc$", text="ABC", flags=["IGNORECASE"])["full_match"]
        is True
    )
    with pytest.raises(ToolError, match="invalid regular expression"):
        run(data.regex_test, pattern="(", text="x")


def test_regex_test_stops_catastrophic_backtracking(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data, "_REGEX_TIMEOUT", 0.5)
    with pytest.raises(ToolError, match="catastrophic backtracking"):
        run(data.regex_test, pattern=r"(a+)+$", text="a" * 40 + "b")


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        ("6*7", "42"),
        ("(1 - 0.999) * 30 * 24 * 60", "43.2"),
        ("2**64 - 1", "18446744073709551615 (18,446,744,073,709,551,615; 0xffffffffffffffff)"),
        ("0xff & 0b1010", "10 (0xa)"),
        ("1/3", "0.333333333333333"),
        ("sqrt(16) + log2(1024)", "14"),
        ("max([3, 9, 4])", "9"),
        ("comb(52, 5)", "2598960 (2,598,960; 0x27a830)"),
        ("1 < 2 < 3", "True"),
        ("10**5000", "≈ 1.000000e5000 (5,001 digits)"),
        ("round(pi, 4)", "3.1416"),
    ],
)
def test_calculate(run: Run, expression: str, result: str) -> None:
    assert run(data.calculate, expression=expression) == f"{expression} = {result}"


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("__import__('os').system('id')", "unsupported expression"),
        ("x + 1", "unknown name 'x'"),
        ("1/0", "division by zero"),
        ("2**10**10", "too large"),
        ("1 <<", "invalid expression"),
        ("factorial(10**6)", "between 0 and 5000"),
        ("sqrt(-1)", "couldn't evaluate"),
        ("(1).__class__", "unsupported expression"),
    ],
)
def test_calculate_rejects(run: Run, expression: str, message: str) -> None:
    with pytest.raises(ToolError, match=re.escape(message)):
        run(data.calculate, expression=expression)


def test_cidr_info(run: Run) -> None:
    info = run(
        data.cidr_info,
        network="10.0.0.5/22",
        check=["10.0.3.17", "10.0.4.1", "10.0.1.0/24", "::1", "bogus"],
        split_prefix=24,
    )
    assert info["network"] == "10.0.0.0/22"
    assert "host bits set" in info["note"]
    assert (info["netmask"], info["broadcast"], info["usable_hosts"]) == (
        "255.255.252.0",
        "10.0.3.255",
        1022,
    )
    assert info["host_range"] == "10.0.0.1 - 10.0.3.254"
    assert info["type"] == ["private"]
    assert info["membership"] == {
        "10.0.3.17": "inside",
        "10.0.4.1": "outside",
        "10.0.1.0/24": "inside",
        "::1": "different IP version (IPv6)",
        "bogus": "invalid address",
    }
    assert info["subnets"] == {
        "count": 4,
        "listed": ["10.0.0.0/24", "10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"],
    }


def test_cidr_info_edge_cases(run: Run) -> None:
    assert run(data.cidr_info, network="192.168.1.10")["usable_hosts"] == 1
    assert run(data.cidr_info, network="10.0.0.0/31")["usable_hosts"] == 2
    v6 = run(data.cidr_info, network="2001:db8::/32")
    assert v6["version"] == "IPv6" and "note" not in v6 and "broadcast" not in v6
    with pytest.raises(ToolError, match="split_prefix must be between 22 and 32"):
        run(data.cidr_info, network="10.0.0.0/22", split_prefix=20)
    with pytest.raises(ToolError, match="invalid network"):
        run(data.cidr_info, network="10.0.0.0/33")


def test_generate_secret(run: Run) -> None:
    tokens = run(data.generate_secret, count=3).splitlines()
    assert len(tokens) == 3 and len(set(tokens)) == 3
    assert len(run(data.generate_secret, kind="hex", length=16)) == 32
    password = run(data.generate_secret, kind="password", length=20)
    assert len(password) == 20
    assert any(c.isupper() for c in password) and any(c.isdigit() for c in password)
    assert uuid.UUID(run(data.generate_secret, kind="uuid4")).version == 4


def test_json_query(run: Run, tmp_path: Path) -> None:
    doc = {
        "items": [
            {"name": "a", "status": {"phase": "Running"}},
            {"name": "b", "status": {"phase": "Failed"}},
        ],
        "kind": "List",
    }
    (tmp_path / "pods.json").write_text(json.dumps(doc))
    assert json.loads(run(data.json_query, path="pods.json", query="items[*].status.phase")) == [
        "Running",
        "Failed",
    ]
    assert json.loads(run(data.json_query, path="pods.json", query="items[-1].name")) == "b"
    assert json.loads(run(data.json_query, path="pods.json", query="$.kind")) == "List"
    assert json.loads(run(data.json_query, text='{"a.b": {"c": 1}}', query='["a.b"].c')) == 1
    assert json.loads(run(data.json_query, text='{"x": 1}\n{"x": 2}\n', query="[*].x")) == [1, 2]
    with pytest.raises(ToolError, match="key 'nope' not found; available keys: items, kind"):
        run(data.json_query, path="pods.json", query="nope")
    with pytest.raises(ToolError, match=r"index \[5\]"):
        run(data.json_query, path="pods.json", query="items[5]")
    with pytest.raises(ToolError, match="exactly one"):
        run(data.json_query, query="x")
    with pytest.raises(ToolError, match="not valid JSON"):
        run(data.json_query, text="{nope")


def test_json_query_outlines_large_results(run: Run) -> None:
    big = {
        "packages": {f"pkg{i}": {"version": "1.0.0", "deps": list(range(50))} for i in range(200)}
    }
    out = run(data.json_query, text=json.dumps(big))
    assert out.startswith("The result is large")
    assert "packages: an object with 200 keys" in out
