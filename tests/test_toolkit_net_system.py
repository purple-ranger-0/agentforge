from __future__ import annotations

import http.server
import json
import os
import shutil
import socket
import ssl
import subprocess
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from agentforge.errors import ToolError
from agentforge.toolkit import net, system

Run = Callable[..., Any]


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        routes = {
            "/json": (200, "application/json", json.dumps({"ok": True, "items": [1, 2]}).encode()),
            "/html": (
                200,
                "text/html; charset=utf-8",
                b"<html><head><title>Docs</title><style>x{}</style></head><body>"
                b"<h1>Hello</h1><p>World &amp; more</p><script>evil()</script></body></html>",
            ),
            "/bin": (200, "application/octet-stream", b"\x00\x01\x02"),
            "/fail": (503, "text/plain", b"try later"),
        }
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/json")
            self.end_headers()
            return
        status, content_type, body = routes.get(self.path, (404, "text/plain", b"missing"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Set-Cookie", "sid=topsecret; Path=/; HttpOnly")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        reply = json.dumps(
            {"got": body.decode(), "type": self.headers.get("Content-Type")}
        ).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_fetch_json(server: str, run: Run) -> None:
    out = run(net.fetch_url, url=f"{server}/json")
    assert out.startswith(f"HTTP 200 OK · GET {server}/json · ")
    assert "content-type: application/json" in out
    assert "set-cookie: sid=<redacted>; Path=/; HttpOnly" in out
    assert "topsecret" not in out
    assert '"items": [\n    1,\n    2\n  ]' in out


def test_fetch_html_is_reduced_to_text(server: str, run: Run) -> None:
    out = run(net.fetch_url, url=f"{server}/html")
    body = out.split("Body (")[1]
    assert "Title: Docs" in body and "Hello\nWorld & more" in body
    assert "evil" not in body and "x{}" not in body


def test_fetch_follows_redirects_and_reports_errors(server: str, run: Run) -> None:
    assert f"Final URL after redirects: {server}/json" in run(
        net.fetch_url, url=f"{server}/redirect"
    )
    failed = run(net.fetch_url, url=f"{server}/fail")
    assert failed.startswith("HTTP 503") and "try later" in failed
    assert "binary application/octet-stream, 3 B, sha256" in run(net.fetch_url, url=f"{server}/bin")
    head = run(net.fetch_url, url=f"{server}/json", method="HEAD")
    assert head.endswith("Body: (empty)")


def test_fetch_rejects_other_schemes_and_dead_hosts(run: Run) -> None:
    with pytest.raises(ToolError, match="only absolute http"):
        run(net.fetch_url, url="file:///etc/passwd")
    with pytest.raises(ToolError, match="failed"):
        run(net.fetch_url, url="http://127.0.0.1:1/", timeout=2)


def test_http_request_sends_json(server: str, run: Run, ctx: Any) -> None:
    out = run(net.http_request, url=f"{server}/items", body='{"name": "x"}')
    assert out.startswith("HTTP 201")
    assert '"got": "{\\"name\\": \\"x\\"}"' in out
    assert '"type": "application/json"' in out
    preview = net.http_request.render_preview(
        ctx, {"url": f"{server}/items", "method": "POST", "body": "{}", "headers": {"X-A": "1"}}
    )
    assert preview.splitlines() == [f"POST {server}/items", "X-A: 1", "", "{}"]


def test_check_ports(run: Run) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    try:
        out = run(net.check_ports, host="127.0.0.1", ports=[port, 1])
    finally:
        listener.close()
    assert f"  {port:>5}/tcp  open" in out
    assert "      1/tcp  closed (connection refused)" in out
    with pytest.raises(ToolError, match="invalid port"):
        run(net.check_ports, host="127.0.0.1", ports=[0])


def test_dns_lookup(run: Run) -> None:
    result = run(net.dns_lookup, name="http://localhost:8080/path")
    assert result["name"] == "localhost"
    assert "127.0.0.1" in result["ipv4"] or "::1" in result["ipv6"]
    with pytest.raises(ToolError, match="couldn't resolve"):
        run(net.dns_lookup, name="no-such-host.invalid")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("example.com", ("example.com", None)),
        ("example.com:8443", ("example.com", 8443)),
        ("https://example.com:8443/x", ("example.com", 8443)),
        ("https://example.com/x", ("example.com", None)),
        ("[::1]:8080", ("::1", 8080)),
        ("::1", ("::1", None)),
    ],
)
def test_split_host(value: str, expected: tuple[str, int | None]) -> None:
    assert net._split_host(value) == expected


@pytest.fixture
def tls_server(tmp_path: Path) -> Iterator[int]:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed")
    key, cert = tmp_path / "key.pem", tmp_path / "cert.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "30",
            "-subj",
            "/CN=agentforge.test",
            "-addext",
            "subjectAltName=DNS:agentforge.test,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    stop = threading.Event()

    def serve() -> None:
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            try:
                with context.wrap_socket(conn, server_side=True):
                    pass
            except (OSError, ssl.SSLError):
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield listener.getsockname()[1]
    stop.set()
    thread.join()
    listener.close()


def test_tls_certificate_of_a_self_signed_server(tls_server: int, run: Run) -> None:
    info = run(net.tls_certificate, host=f"127.0.0.1:{tls_server}")
    assert info["port"] == tls_server
    assert info["verified"] is False
    assert "self" in info["verify_error"].lower()
    assert info["subject"] == "CN=agentforge.test"
    assert info["subject_alt_names"] == ["agentforge.test", "127.0.0.1"]
    assert 28 <= info["days_remaining"] <= 30
    assert info["tls_version"].startswith("TLS")


# ----------------------------------------------------------------------------- system


def test_system_info(run: Run, tmp_path: Path) -> None:
    info = run(system.system_info)
    assert info["cpu_cores"] >= 1
    assert info["workspace_root"] == str(tmp_path)
    assert info["disks"][0]["path"] == str(tmp_path)
    assert {"os", "hostname", "python", "time"} <= set(info)


@pytest.mark.skipif(os.name == "nt" or shutil.which("ps") is None, reason="needs ps")
def test_list_processes_finds_this_test_run(run: Run) -> None:
    out = run(system.list_processes, name_contains="pytest", show_args=True, sort_by="memory")
    assert out.splitlines()[0].split() == [
        "PID",
        "PPID",
        "USER",
        "%CPU",
        "%MEM",
        "RSS",
        "ELAPSED",
        "COMMAND",
    ]
    assert any(line.split()[0] == str(os.getpid()) for line in out.splitlines()[1:-1])
    assert "matching 'pytest'" in out.splitlines()[-1]


@pytest.mark.skipif(os.name == "nt", reason="POSIX tools")
def test_listening_ports(run: Run) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    try:
        out = run(system.listening_ports, port=port)
    finally:
        listener.close()
    assert out.startswith("$ ")
    assert f":{port}" in out
    assert run(system.listening_ports, port=1).startswith("Nothing is listening on port 1")


def test_proc_address_decoding() -> None:
    assert system._proc_address("0100007F") == "127.0.0.1"
    assert system._proc_address("00000000000000000000000001000000") == "[::1]"


def test_which(run: Run) -> None:
    result = run(system.which, programs=["sh", "definitely-not-a-real-program"])
    assert result["definitely-not-a-real-program"] == "not found"
    assert result["sh"].endswith("sh")
    with pytest.raises(ToolError):
        run(system.which, programs=[])
