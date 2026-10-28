"""Network tools: HTTP requests, DNS lookups, TCP port checks and TLS certificates."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import html.parser
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any, Literal

from .._version import __version__
from ..errors import ToolError
from ..tools import ToolContext, tool
from ._util import human_size, relative_time

_USER_AGENT = f"agentforge/{__version__}"
_MAX_BODY_BYTES = 5 * 1024 * 1024
_TEXTUAL_TYPES = (
    "text/",
    "json",
    "xml",
    "javascript",
    "yaml",
    "csv",
    "x-www-form-urlencoded",
    "toml",
)
_SET_COOKIE_VALUE = re.compile(r"^([^=;]+)=[^;]*")


def _check_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ToolError("only absolute http:// and https:// URLs are supported")
    return url.strip()


def _request(
    url: str,
    method: str,
    headers: Mapping[str, str] | None,
    body: str | None,
    timeout: float,
    max_chars: int,
) -> str:
    url = _check_url(url)
    data = body.encode("utf-8") if body is not None else None
    request_headers = {"User-Agent": _USER_AGENT, "Accept": "*/*", **(headers or {})}
    if data is not None and not any(k.lower() == "content-type" for k in request_headers):
        request_headers["Content-Type"] = (
            "application/json" if _is_json(body or "") else "text/plain; charset=utf-8"
        )
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    timeout = max(0.5, min(timeout, 120.0))
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc  # 4xx/5xx responses still have a status, headers and a body
    except urllib.error.URLError as exc:
        raise ToolError(f"{method} {url} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ToolError(f"{method} {url} timed out after {timeout:g}s") from exc
    except (ValueError, http.client.HTTPException) as exc:
        raise ToolError(f"{method} {url} failed: {exc}") from exc
    try:
        with response:
            raw = response.read(_MAX_BODY_BYTES + 1) if method != "HEAD" else b""
            status = getattr(response, "status", None) or response.getcode()
            reason = response.reason or ""
            final_url = response.geturl()
            header_items = list(response.headers.items())
            content_type = response.headers.get("Content-Type", "")
            charset = response.headers.get_content_charset() or "utf-8"
    except TimeoutError as exc:
        raise ToolError(f"{method} {url}: timed out while reading the response") from exc
    elapsed_ms = (time.monotonic() - started) * 1000

    lines = [f"HTTP {status} {reason} · {method} {url} · {elapsed_ms:.0f} ms"]
    if final_url and final_url != url:
        lines.append(f"Final URL after redirects: {final_url}")
    lines.append("Headers:")
    for name, value in header_items[:60]:
        if name.lower() == "set-cookie":
            value = _SET_COOKIE_VALUE.sub(r"\1=<redacted>", value)
        lines.append(f"  {name.lower()}: {value}")
    lines.append(_format_body(raw, content_type, charset, max_chars))
    return "\n".join(lines)


def _is_json(text: str) -> bool:
    stripped = text.strip()
    if stripped[:1] not in ("{", "["):
        return False
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


def _format_body(raw: bytes, content_type: str, charset: str, max_chars: int) -> str:
    if not raw:
        return "Body: (empty)"
    truncated = len(raw) > _MAX_BODY_BYTES
    raw = raw[:_MAX_BODY_BYTES]
    kind = content_type.split(";")[0].strip().lower()
    textual = any(t in kind for t in _TEXTUAL_TYPES) or (not kind and b"\0" not in raw[:1024])
    if not textual:
        digest = hashlib.sha256(raw).hexdigest()
        return f"Body: binary {kind or 'data'}, {human_size(len(raw))}, sha256 {digest}"
    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    if "json" in kind and not truncated:
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except ValueError:
            pass
    elif "html" in kind:
        text = _html_to_text(text)
    note = ""
    if len(text) > max_chars:
        note = f"\n[... {len(text) - max_chars:,} more characters; raise max_chars to see more]"
        text = text[:max_chars]
    elif truncated:
        note = f"\n[... body truncated at {human_size(_MAX_BODY_BYTES)}]"
    shown = "text from HTML" if "html" in kind else kind or "text"
    return f"Body ({human_size(len(raw))}, {shown}):\n{text}{note}"


class _TextExtractor(html.parser.HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "svg", "template"})
    _BLOCK = frozenset(
        {
            "p",
            "div",
            "br",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "pre",
            "section",
            "article",
            "header",
            "footer",
            "table",
            "ul",
            "ol",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skipping = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skipping:
            self.parts.append(data)


def _html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # malformed markup: fall back to the raw text
        return markup
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(parser.parts))
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    title = parser.title.strip()
    return f"Title: {title}\n\n{text}" if title else text


@tool(risk="network")
def fetch_url(
    ctx: ToolContext,
    url: str,
    method: Literal["GET", "HEAD"] = "GET",
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
    max_chars: int = 20000,
) -> str:
    """Fetch a URL with GET or HEAD and show the status, headers, timing and body.

    Use it to check whether an endpoint or site is up, inspect an API response, read a raw
    file or docs page, or debug redirects, caching and CORS headers. JSON is
    pretty-printed and HTML is reduced to its text. Only http(s) URLs.

    Args:
        url: The URL to fetch.
        method: GET, or HEAD for headers only.
        headers: Extra request headers, e.g. {"Accept": "application/json"}.
        timeout: Seconds to wait for the server.
        max_chars: Maximum characters of body text to return.
    """
    return _request(url, method, headers, None, timeout, max(500, max_chars))


def _request_preview(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    lines = [f"{args.get('method', 'POST')} {args['url']}"]
    for name, value in (args.get("headers") or {}).items():
        lines.append(f"{name}: {value}")
    body = args.get("body")
    if body:
        lines.append("")
        lines.append(
            body if len(body) <= 1500 else body[:1500] + f"\n... ({len(body):,} chars total)"
        )
    return "\n".join(lines)


@tool(risk="exec", preview=_request_preview)
def http_request(
    ctx: ToolContext,
    url: str,
    method: Literal["POST", "PUT", "PATCH", "DELETE", "GET", "HEAD", "OPTIONS"] = "POST",
    body: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_chars: int = 20000,
) -> str:
    """Send an HTTP request that may change something on the server (POST, PUT, PATCH,
    DELETE), e.g. calling a webhook or an API. Always needs the user's approval; for plain
    reads use fetch_url instead. A JSON body gets Content-Type: application/json
    automatically.

    Args:
        url: The URL to send the request to.
        method: HTTP method.
        body: Request body (JSON or plain text).
        headers: Extra request headers, e.g. {"Authorization": "Bearer ..."}.
        timeout: Seconds to wait for the server.
        max_chars: Maximum characters of response body to return.
    """
    return _request(url, method, headers, body, timeout, max(500, max_chars))


def _split_host(value: str) -> tuple[str, int | None]:
    """Accept 'example.com', 'example.com:8443', '[::1]:8080' or a URL; return (host, port)."""
    value = value.strip()
    if "://" in value:
        parts = urllib.parse.urlsplit(value)
        try:
            port = parts.port
        except ValueError:
            port = None
        return parts.hostname or value, port
    if value.startswith("[") and "]" in value:
        host, _, rest = value[1:].partition("]")
        port_text = rest[1:] if rest.startswith(":") else ""
        return host, int(port_text) if port_text.isdigit() else None
    if value.count(":") == 1:  # host:port (a bare IPv6 address has several colons)
        host, _, port_text = value.partition(":")
        return host, int(port_text) if port_text.isdigit() else None
    return value, None


def _host(value: str) -> str:
    return _split_host(value)[0]


@tool(risk="network")
def dns_lookup(ctx: ToolContext, name: str) -> dict[str, Any]:
    """Resolve a host name to its IPv4/IPv6 addresses, or an IP address back to a name,
    using the system resolver. Use it to debug "could not resolve host" errors or to
    check where a domain points.

    Args:
        name: Host name, URL or IP address.
    """
    host = _host(name)
    started = time.monotonic()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        try:
            hostname, aliases, _ = socket.gethostbyaddr(str(address))
        except (socket.herror, socket.gaierror) as exc:
            raise ToolError(f"no reverse DNS name for {address}: {exc}") from exc
        return {"address": str(address), "hostname": hostname, "aliases": aliases}
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME)
    except (socket.gaierror, UnicodeError) as exc:
        raise ToolError(f"couldn't resolve {host}: {exc}") from exc
    return {
        "name": host,
        "canonical_name": next((info[3] for info in infos if info[3]), host),
        "ipv4": sorted({info[4][0] for info in infos if info[0] == socket.AF_INET}),
        "ipv6": sorted({info[4][0] for info in infos if info[0] == socket.AF_INET6}),
        "resolved_in_ms": round((time.monotonic() - started) * 1000, 1),
    }


@tool(risk="network")
def check_ports(ctx: ToolContext, host: str, ports: list[int], timeout: float = 3.0) -> str:
    """Test TCP connections to one or more ports on a host (like `nc -z`), reporting open,
    closed (refused) or no response for each, with connect latency. Use it to check
    whether a service is reachable or a firewall is in the way.

    Args:
        host: Host name or IP address (localhost works).
        ports: Ports to test, e.g. [22, 80, 443, 5432]. At most 100.
        timeout: Seconds to wait for each connection.
    """
    host = _host(host)
    if not ports:
        raise ToolError("give at least one port")
    if len(ports) > 100:
        raise ToolError("check at most 100 ports per call")
    bad = [p for p in ports if not 1 <= p <= 65535]
    if bad:
        raise ToolError(f"invalid port number(s): {bad}")
    timeout = max(0.1, min(timeout, 30.0))
    try:
        address = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)[0][4][0]
    except (socket.gaierror, UnicodeError) as exc:
        raise ToolError(f"couldn't resolve {host}: {exc}") from exc

    def probe(port: int) -> str:
        started = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return f"open  ({(time.monotonic() - started) * 1000:.1f} ms)"
        except ConnectionRefusedError:
            return "closed (connection refused)"
        except TimeoutError:
            return f"no response within {timeout:g}s (filtered, or the host is down)"
        except OSError as exc:
            return f"error: {exc.strerror or exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(ports))) as pool:
        states = list(pool.map(probe, ports))
    lines = [f"{host} ({address}):"]
    lines += [f"  {port:>5}/tcp  {state}" for port, state in zip(ports, states, strict=True)]
    return "\n".join(lines)


_NAME_KEYS = {
    "commonName": "CN",
    "organizationName": "O",
    "organizationalUnitName": "OU",
    "countryName": "C",
    "stateOrProvinceName": "ST",
    "localityName": "L",
}


def _x509_name(name: Any) -> str:
    return ", ".join(f"{_NAME_KEYS.get(k, k)}={v}" for rdn in name for k, v in rdn)


def _cert_time(text: str) -> dt.datetime:
    return dt.datetime.fromtimestamp(ssl.cert_time_to_seconds(text), dt.timezone.utc)


def _unverified_certificate(
    host: str, port: int, timeout: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with (
        socket.create_connection((host, port), timeout=timeout) as sock,
        context.wrap_socket(sock, server_hostname=host) as tls,
    ):
        der = tls.getpeercert(binary_form=True)
        extra = {"tls_version": tls.version(), "cipher": (tls.cipher() or ("?",))[0]}
    decode = getattr(getattr(ssl, "_ssl", None), "_test_decode_cert", None)
    if not der or decode is None:
        return {}, extra
    with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as fh:
        fh.write(ssl.DER_cert_to_PEM_cert(der))
    try:
        return decode(fh.name), extra
    finally:
        os.unlink(fh.name)


@tool(risk="network")
def tls_certificate(
    ctx: ToolContext, host: str, port: int = 443, timeout: float = 10.0
) -> dict[str, Any]:
    """Inspect the TLS certificate a server presents: whether it verifies, subject, issuer,
    alternative names, validity dates and days until expiry, plus the negotiated TLS
    version and cipher. Use it for expiring or invalid certificates and handshake errors.

    Args:
        host: Host name or URL (a port given here, as in "example.com:8443", wins).
        port: TLS port.
        timeout: Seconds to wait for the connection.
    """
    host, given_port = _split_host(host)
    port = given_port or port
    info: dict[str, Any] = {"host": host, "port": port}
    try:
        with (
            socket.create_connection((host, port), timeout=timeout) as sock,
            ssl.create_default_context().wrap_socket(sock, server_hostname=host) as tls,
        ):
            cert = tls.getpeercert() or {}
            info.update(
                verified=True, tls_version=tls.version(), cipher=(tls.cipher() or ("?",))[0]
            )
    except ssl.SSLCertVerificationError as exc:
        info.update(verified=False, verify_error=exc.verify_message or str(exc))
        try:
            cert, extra = _unverified_certificate(host, port, timeout)
        except (OSError, ssl.SSLError) as inner:
            raise ToolError(
                f"certificate didn't verify ({info['verify_error']}) and couldn't be read: {inner}"
            ) from inner
        info.update(extra)
    except (OSError, ssl.SSLError) as exc:
        raise ToolError(f"TLS connection to {host}:{port} failed: {exc}") from exc
    if not cert:
        info["note"] = "the certificate details couldn't be decoded"
        return info
    info["subject"] = _x509_name(cert.get("subject", ()))
    info["issuer"] = _x509_name(cert.get("issuer", ()))
    info["subject_alt_names"] = [value for _, value in cert.get("subjectAltName", ())]
    if "notBefore" in cert and "notAfter" in cert:
        not_before, not_after = _cert_time(cert["notBefore"]), _cert_time(cert["notAfter"])
        now = dt.datetime.now(dt.timezone.utc)
        info["not_before"] = not_before.isoformat()
        info["not_after"] = not_after.isoformat()
        info["expires"] = relative_time(not_after, now)
        info["days_remaining"] = (not_after - now).days
    if "serialNumber" in cert:
        info["serial_number"] = cert["serialNumber"]
    return info


TOOLS = (fetch_url, http_request, dns_lookup, check_ports, tls_certificate)
