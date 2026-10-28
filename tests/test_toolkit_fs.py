from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agentforge import ToolContext
from agentforge.errors import ToolError
from agentforge.toolkit import fs
from agentforge.toolkit._util import glob_match, iter_files


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "main.py").write_text(
        "import os\n\ndef main():\n    print('hello')\n    # TODO: config\n"
    )
    (tmp_path / "src" / "app" / "util.py").write_text(
        "def helper():\n    return 42  # todo later\n"
    )
    (tmp_path / "src" / "web.ts").write_text("export const x = 1;\n")
    (tmp_path / "README.md").write_text("# Demo\n")
    (tmp_path / ".env.example").write_text("KEY=\n")
    (tmp_path / "node_modules" / "dep").mkdir(parents=True)
    (tmp_path / "node_modules" / "dep" / "index.js").write_text("// TODO in a dependency\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    (tmp_path / "log.txt").write_text("".join(f"line {i}\n" for i in range(1, 1001)))
    return tmp_path


Run = Callable[..., Any]


def test_read_file_pages_and_numbers(project: Path, run: Run) -> None:
    out = run(fs.read_file, path="log.txt", start_line=10, max_lines=3)
    assert out.splitlines() == [
        "log.txt (lines 10-12 of 1000)",
        "    10\tline 10",
        "    11\tline 11",
        "    12\tline 12",
        "[988 more lines; continue with start_line=13]",
    ]


def test_read_file_tail(project: Path, run: Run) -> None:
    out = run(fs.read_file, path="log.txt", start_line=-2)
    assert out.splitlines() == [
        "log.txt (lines 999-1000 of 1000)",
        "   999\tline 999",
        "  1000\tline 1000",
    ]


def test_read_file_edge_cases(project: Path, run: Run) -> None:
    (project / "empty.txt").write_text("")
    assert run(fs.read_file, path="empty.txt") == "empty.txt is empty."
    assert "past the end" in run(fs.read_file, path="README.md", start_line=50)
    with pytest.raises(ToolError, match="binary"):
        run(fs.read_file, path="logo.png")
    (project / "wide.txt").write_text("x" * 5000)
    assert "[+3000 chars]" in run(fs.read_file, path="wide.txt")


def test_list_dir(project: Path, run: Run) -> None:
    out = run(fs.list_dir, depth=2)
    lines = out.splitlines()
    assert lines[0] == "./"
    assert "  node_modules/ (1 entries)" in lines  # summarized, not expanded
    assert "  src/" in lines
    assert "    app/ (2 entries)" in lines
    assert any(line.startswith("  .env.example") for line in lines)
    hidden_off = run(fs.list_dir, show_hidden=False)
    assert ".env.example" not in hidden_off


def test_list_dir_truncates(tmp_path: Path, run: Run) -> None:
    for i in range(600):
        (tmp_path / f"f{i:03}.txt").write_text("")
    assert "listing truncated at 500 entries" in run(fs.list_dir)


def test_find_files(project: Path, run: Run) -> None:
    out = run(fs.find_files, pattern="*.py")
    assert out.splitlines()[1:] == ["src/app/main.py", "src/app/util.py"]
    assert "src/web.ts" in run(fs.find_files, pattern="src/*.{ts,tsx}")
    assert "No files matching" in run(fs.find_files, pattern="*.go")
    assert "node_modules" not in run(fs.find_files, pattern="*.js")


def test_search_text(project: Path, run: Run) -> None:
    out = run(fs.search_text, pattern="todo", ignore_case=True)
    assert out.splitlines() == [
        "2 matching line(s) in 2 file(s):",
        "src/app/main.py:5:    # TODO: config",
        "src/app/util.py:2:    return 42  # todo later",
    ]
    assert "util.py" not in run(fs.search_text, pattern="TODO", glob="main.py")
    context = run(fs.search_text, pattern="print", context_lines=1)
    assert "src/app/main.py-3-def main():" in context
    assert "src/app/main.py:4:    print('hello')" in context
    assert "No matches" in run(fs.search_text, pattern="nothing-like-this")
    assert "1 matching line" in run(fs.search_text, pattern="42  #", literal=True)
    with pytest.raises(ToolError, match="invalid regular expression"):
        run(fs.search_text, pattern="(unclosed")


def test_search_text_respects_max_results(project: Path, run: Run) -> None:
    out = run(fs.search_text, pattern="line", path="log.txt", max_results=5)
    assert out.splitlines()[0] == "5 matching line(s) in 1 file(s) (stopped at max_results=5):"


def test_search_excerpts_long_lines_around_the_match(tmp_path: Path, run: Run) -> None:
    (tmp_path / "min.js").write_text("a" * 2000 + "NEEDLE" + "b" * 2000)
    line = run(fs.search_text, pattern="NEEDLE").splitlines()[1]
    assert "NEEDLE" in line
    assert len(line) < 400


def test_file_info(project: Path, run: Run) -> None:
    info = run(fs.file_info, path="src/app/main.py")
    assert info["type"] == "file"
    assert info["lines"] == 5
    assert info["binary"] is False
    assert info["permissions"].startswith("-r")
    assert run(fs.file_info, path="logo.png")["binary"] is True
    assert run(fs.file_info, path="src")["entries"] == 2


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_file_info_reports_symlinks(project: Path, run: Run) -> None:
    (project / "link.md").symlink_to(project / "README.md")
    assert run(fs.file_info, path="link.md")["symlink_target"] == str(project / "README.md")


def test_disk_usage(project: Path, run: Run) -> None:
    (project / "big.bin").write_bytes(b"0" * 50_000)
    out = run(fs.disk_usage, top=3)
    assert "Filesystem holding ." in out
    entries = out.split("Largest entries:")[1].split("Largest files:")[0]
    assert entries.strip().splitlines()[0].endswith("big.bin")
    assert "big.bin" in out.split("Largest files:")[1]


def test_diff_files(project: Path, run: Run) -> None:
    (project / "a.txt").write_text("one\ntwo\n")
    (project / "b.txt").write_text("one\nthree\n")
    out = run(fs.diff_files, path_a="a.txt", path_b="b.txt")
    assert out.splitlines()[-2:] == ["-two", "+three"]
    assert "identical" in run(fs.diff_files, path_a="a.txt", path_b="a.txt")


def test_write_file(tmp_path: Path, run: Run, ctx: ToolContext) -> None:
    assert (
        run(fs.write_file, path="new/dir/f.txt", content="a\nb")
        == "Created new/dir/f.txt (2 lines, 3 B)."
    )
    assert (tmp_path / "new/dir/f.txt").read_text() == "a\nb"
    preview = fs.write_file.render_preview(ctx, {"path": "new/dir/f.txt", "content": "a\nc"})
    assert "-b" in preview and "+c" in preview
    assert run(fs.write_file, path="new/dir/f.txt", content="x\n").startswith("Overwrote")
    with pytest.raises(ToolError, match="is a directory"):
        run(fs.write_file, path="new", content="x")
    new_preview = fs.write_file.render_preview(ctx, {"path": "fresh.txt", "content": "1\n2\n"})
    assert new_preview.startswith("new file fresh.txt (2 lines)")


def test_edit_file(tmp_path: Path, run: Run, ctx: ToolContext) -> None:
    target = tmp_path / "conf.ini"
    target.write_text("port=1\nhost=a\nport=1\n")
    with pytest.raises(ToolError, match="appears 2 times"):
        run(fs.edit_file, path="conf.ini", old_text="port=1", new_text="port=2")
    with pytest.raises(ToolError, match="wasn't found"):
        run(fs.edit_file, path="conf.ini", old_text="nope", new_text="x")
    with pytest.raises(ToolError, match="empty"):
        run(fs.edit_file, path="conf.ini", old_text="", new_text="x")
    preview = fs.edit_file.render_preview(
        ctx, {"path": "conf.ini", "old_text": "host=a", "new_text": "host=b"}
    )
    assert "-host=a" in preview and "+host=b" in preview
    assert target.read_text() == "port=1\nhost=a\nport=1\n"  # previews don't write
    out = run(fs.edit_file, path="conf.ini", old_text="port=1", new_text="port=2", replace_all=True)
    assert out.startswith("Edited conf.ini (2 replacements).")
    assert target.read_text() == "port=2\nhost=a\nport=2\n"


def test_edit_file_matches_windows_line_endings(tmp_path: Path, run: Run) -> None:
    target = tmp_path / "win.txt"
    target.write_bytes(b"a\r\nb\r\nc\r\n")
    run(fs.edit_file, path="win.txt", old_text="a\nb", new_text="a\nB")
    assert target.read_bytes() == b"a\r\nB\r\nc\r\n"


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("src/app/main.py", "*.py", True),
        ("main.py", "*.py", True),
        ("src/app/main.py", "src/*.py", False),
        ("src/app/main.py", "src/**/*.py", True),
        ("src/main.py", "src/**/*.py", True),
        ("src/app/main.py", "**/app/*.py", True),
        ("a.yml", "*.{yml,yaml}", True),
        ("a.yaml", "*.{yml,yaml}", True),
        ("a.json", "*.{yml,yaml}", False),
        ("Dockerfile.dev", "Dockerfile*", True),
        ("x1.txt", "x?.txt", True),
        ("xa.txt", "x[0-9].txt", False),
        ("xa.txt", "x[!0-9].txt", True),
        ("src/web.ts", "./src/*.ts", True),
    ],
)
def test_glob_match(path: str, pattern: str, expected: bool) -> None:
    assert glob_match(path, pattern) is expected


def test_iter_files_respects_gitignore(tmp_path: Path, git_repo: Callable[..., str]) -> None:
    (tmp_path / ".gitignore").write_text("build/\n*.log\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.js").write_text("")
    (tmp_path / "debug.log").write_text("")
    (tmp_path / "keep.py").write_text("")
    names = [p.name for p in iter_files(tmp_path)]
    assert names == [".gitignore", "keep.py"]
