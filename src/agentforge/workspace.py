"""The directory that file-system tools are confined to."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from .errors import ToolError

PathKind = Literal["file", "dir", "exists"]


class Workspace:
    """A root directory; every path a tool touches must resolve inside it.

    Paths are resolved with symlinks followed, so a link pointing outside the root is
    rejected just like ``../../etc/passwd``.
    """

    def __init__(self, root: str | os.PathLike[str] = ".") -> None:
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"workspace root is not a directory: {path}")
        self.root = path

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"

    def resolve(self, path: str | os.PathLike[str], kind: PathKind | None = None) -> Path:
        """Resolve ``path`` (absolute, or relative to the root) and check it stays inside the root.

        ``kind`` additionally requires the path to exist as a file, a directory, or anything.
        """
        raw = os.fspath(path)
        if not raw.strip():
            raise ToolError("path is empty")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if not self.contains(resolved):
            raise ToolError(
                f"{raw!r} is outside the workspace root ({self.root}). Tools can only access "
                "paths under the root; the user can pick a different one with --root."
            )
        if kind == "file":
            if not resolved.exists():
                raise ToolError(f"no such file: {raw}")
            if not resolved.is_file():
                raise ToolError(f"not a regular file: {raw}")
        elif kind == "dir":
            if not resolved.exists():
                raise ToolError(f"no such directory: {raw}")
            if not resolved.is_dir():
                raise ToolError(f"not a directory: {raw}")
        elif kind == "exists" and not resolved.exists():
            raise ToolError(f"no such file or directory: {raw}")
        return resolved

    def contains(self, path: Path) -> bool:
        return path == self.root or path.is_relative_to(self.root)

    def display(self, path: Path) -> str:
        """Path relative to the root (for output), or absolute if it lies outside."""
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return str(path)
        text = relative.as_posix()
        return text if text != "." else "."
