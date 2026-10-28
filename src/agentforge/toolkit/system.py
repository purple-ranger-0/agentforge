"""Tools that describe the local machine: OS and resources, processes, listening ports, programs."""

from __future__ import annotations

import datetime as dt
import getpass
import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Literal

from ..errors import ToolError
from ..tools import ToolContext, tool
from ._util import clamp, format_duration, human_size, run


@tool
def system_info(ctx: ToolContext) -> dict[str, Any]:
    """Summarize this machine: OS and distribution, CPU, memory, load, uptime, disk space,
    user, shell, time zone, Python version and whether it runs in a container. Use it to
    orient yourself before diagnosing a machine-specific problem."""
    uname = platform.uname()
    info: dict[str, Any] = {
        "os": f"{uname.system} {uname.release}".strip(),
        "distribution": _distribution(),
        "architecture": uname.machine,
        "hostname": socket.gethostname(),
        "user": _user(),
        "cpu": _cpu_model(),
        "cpu_cores": os.cpu_count(),
    }
    if hasattr(os, "getloadavg"):
        try:
            info["load_average_1_5_15m"] = [round(x, 2) for x in os.getloadavg()]
        except OSError:
            pass
    info["memory"] = _memory()
    uptime = _uptime_seconds()
    if uptime is not None:
        info["uptime"] = format_duration(uptime)
    info["disks"] = _disks(ctx.workspace.root)
    now = dt.datetime.now().astimezone()
    info["time"] = now.isoformat(timespec="seconds")
    info["timezone"] = now.tzname()
    info["shell"] = os.environ.get("SHELL") or os.environ.get("COMSPEC")
    info["python"] = f"{platform.python_implementation()} {platform.python_version()}"
    info["container"] = _container()
    info["workspace_root"] = str(ctx.workspace.root)
    return {k: v for k, v in info.items() if v not in (None, "", {})}


def _user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:
        return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _distribution() -> str | None:
    os_release = _read("/etc/os-release")
    if os_release:
        match = re.search(r'^PRETTY_NAME="?([^"\n]*)"?', os_release, re.MULTILINE)
        if match:
            return match.group(1)
    if platform.system() == "Darwin":
        version = platform.mac_ver()[0]
        return f"macOS {version}" if version else "macOS"
    if platform.system() == "Windows":
        return f"Windows {platform.version()}"
    return None


def _cpu_model() -> str | None:
    cpuinfo = _read("/proc/cpuinfo")
    if cpuinfo:
        match = re.search(r"^(?:model name|Hardware|cpu model)\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if match:
            return match.group(1).strip()
    if platform.system() == "Darwin" and shutil.which("sysctl"):
        proc = run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return platform.processor() or None


def _memory() -> dict[str, str]:
    meminfo = _read("/proc/meminfo")
    if meminfo:
        values = {
            key: int(amount) * 1024
            for key, amount in re.findall(r"^(\w+):\s+(\d+) kB", meminfo, re.MULTILINE)
        }
        memory: dict[str, str] = {}
        if "MemTotal" in values:
            total = values["MemTotal"]
            memory["total"] = human_size(total)
            if "MemAvailable" in values:
                available = values["MemAvailable"]
                memory["available"] = human_size(available)
                memory["used_percent"] = f"{(total - available) / total:.0%}"
        if values.get("SwapTotal"):
            memory["swap_total"] = human_size(values["SwapTotal"])
            memory["swap_free"] = human_size(values.get("SwapFree", 0))
        return memory
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return {}
    return {"total": human_size(total)}


def _uptime_seconds() -> float | None:
    uptime = _read("/proc/uptime")
    if uptime:
        try:
            return float(uptime.split()[0])
        except (IndexError, ValueError):
            return None
    if platform.system() == "Darwin" and shutil.which("sysctl"):
        proc = run(["sysctl", "-n", "kern.boottime"], timeout=5)
        match = re.search(r"sec = (\d+)", proc.stdout)
        if match:
            return dt.datetime.now().timestamp() - int(match.group(1))
    return None


def _disks(workspace_root: Path) -> list[dict[str, str]]:
    disks = []
    seen = set()
    for path in (workspace_root, Path(workspace_root.anchor or "/")):
        try:
            device = path.stat().st_dev
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        if device in seen:
            continue
        seen.add(device)
        disks.append(
            {
                "path": str(path),
                "total": human_size(usage.total),
                "used": human_size(usage.used),
                "free": human_size(usage.free),
                "used_percent": f"{usage.used / usage.total:.0%}" if usage.total else "?",
            }
        )
    return disks


def _container() -> str | None:
    if os.path.exists("/.dockerenv"):
        return "docker"
    if os.path.exists("/run/.containerenv"):
        return "podman"
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    cgroup = _read("/proc/1/cgroup") or ""
    for marker in ("kubepods", "docker", "containerd", "lxc"):
        if marker in cgroup:
            return marker
    return None


_SORT_KEYS = {"cpu": (3, True), "memory": (5, True), "pid": (0, False)}
_ROW = "{:>7} {:>7} {:<12} {:>5} {:>5} {:>9}  {:>11}  {}"


@tool
def list_processes(
    ctx: ToolContext,
    sort_by: Literal["cpu", "memory", "pid"] = "cpu",
    limit: int = 20,
    name_contains: str | None = None,
    show_args: bool = False,
) -> str:
    """List running processes with PID, parent PID, user, CPU %, memory and elapsed time
    (like `ps aux`), sorted and filtered. Use it for "what's eating my CPU/memory?" or to
    find a process's PID.

    Args:
        sort_by: Sort by CPU %, resident memory, or PID.
        limit: Maximum number of processes to show (1-500).
        name_contains: Only processes whose command contains this text (case-insensitive).
        show_args: Show full command lines instead of executable names.
    """
    if os.name == "nt":
        raise ToolError(
            "list_processes needs ps, which Windows doesn't have; try run_command with `tasklist`"
        )
    if shutil.which("ps") is None:
        raise ToolError("ps isn't installed")
    column = "args" if show_args else "comm"
    proc = run(
        ["ps", "-A", "-ww", "-o", f"pid=,ppid=,user=,%cpu=,%mem=,rss=,etime=,{column}="], timeout=15
    )
    if proc.returncode != 0:
        raise ToolError(f"ps failed: {proc.stderr.strip()}")
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        pid, ppid, user, cpu, mem, rss, elapsed, command = parts
        try:
            rows.append(
                (
                    int(pid),
                    int(ppid),
                    user,
                    float(cpu),
                    float(mem),
                    int(rss) * 1024,
                    elapsed,
                    command,
                )
            )
        except ValueError:
            continue
    total = len(rows)
    if name_contains:
        needle = name_contains.lower()
        rows = [r for r in rows if needle in r[7].lower()]
    index, reverse = _SORT_KEYS[sort_by]
    rows.sort(key=lambda r: r[index], reverse=reverse)
    limit = clamp(limit, 1, 500)
    lines = [_ROW.format("PID", "PPID", "USER", "%CPU", "%MEM", "RSS", "ELAPSED", "COMMAND")]
    for pid, ppid, user, cpu, mem, rss, elapsed, command in rows[:limit]:
        if len(command) > 200:
            command = command[:200] + "..."
        lines.append(
            _ROW.format(
                pid, ppid, user[:12], f"{cpu:.1f}", f"{mem:.1f}", human_size(rss), elapsed, command
            )
        )
    shown = min(limit, len(rows))
    filtered = f", {len(rows)} matching {name_contains!r}" if name_contains else ""
    note = f"({total} processes{filtered}; showing {shown} by {sort_by}"
    if platform.system() == "Linux" and sort_by == "cpu":
        note += "; on Linux %CPU is averaged over each process's lifetime"
    return "\n".join([*lines, note + ")"])


@tool
def listening_ports(ctx: ToolContext, port: int | None = None) -> str:
    """Show which TCP/UDP ports are listening on this machine and, where the OS allows,
    which process owns each (like `ss -ltnup` or `lsof -iTCP -sTCP:LISTEN`). Use it for
    "what's running on port 8080?" or "is my server actually listening?".

    Args:
        port: Only show this port.
    """
    commands = []
    if shutil.which("ss"):
        commands.append(["ss", "-ltnup"])
    if shutil.which("lsof"):
        commands.append(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"])
    if shutil.which("netstat"):
        commands.append(["netstat", "-an"])
    output, used = "", ""
    for cmd in commands:
        try:
            proc = run(cmd, timeout=15)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            output, used = proc.stdout, " ".join(cmd)
            break
    if not output:
        output, used = _proc_net_listeners(), "/proc/net"
    if not output:
        raise ToolError("couldn't list listening ports: none of ss, lsof or netstat is available")
    lines = output.rstrip("\n").splitlines()
    header, rows = lines[0], lines[1:]
    if used.startswith("netstat"):
        rows = [r for r in rows if "LISTEN" in r]
    if port is not None:
        # ":8080 " (ss, lsof, Linux netstat) or ".8080 " (BSD netstat); never part of an IP.
        pattern = re.compile(rf"[:.]{port}(?=\s|$)")
        rows = [r for r in rows if pattern.search(r)]
        if not rows:
            return f"Nothing is listening on port {port} (checked with {used})."
    note = ""
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        note = "\n(Process names are only shown for processes owned by the current user.)"
    return f"$ {used}\n" + "\n".join([header, *rows]) + note


def _proc_net_listeners() -> str:
    rows = []
    for name in ("tcp", "tcp6"):
        content = _read(f"/proc/net/{name}")
        if not content:
            continue
        for line in content.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":  # 0A = LISTEN
                continue
            address, port_hex = fields[1].rsplit(":", 1)
            rows.append(f"{name:<5} {_proc_address(address)}:{int(port_hex, 16)}")
    return "Proto Local address\n" + "\n".join(rows) if rows else ""


def _proc_address(hex_address: str) -> str:
    raw = bytes.fromhex(hex_address)
    if len(raw) == 4:
        return socket.inet_ntop(socket.AF_INET, raw[::-1])
    words = b"".join(raw[i : i + 4][::-1] for i in range(0, 16, 4))
    return "[" + socket.inet_ntop(socket.AF_INET6, words) + "]"


@tool
def which(ctx: ToolContext, programs: list[str]) -> dict[str, str]:
    """Check whether command-line programs are installed and where (like `which`).
    Use it before suggesting or running commands that need tools like docker, kubectl,
    node, terraform or psql.

    Args:
        programs: Program names, e.g. ["docker", "kubectl"].
    """
    if not programs:
        raise ToolError("give at least one program name")
    return {name: shutil.which(name) or "not found" for name in programs[:50]}


TOOLS = (system_info, list_processes, listening_ports, which)
