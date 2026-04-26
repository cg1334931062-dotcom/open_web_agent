import asyncio
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MAX_FILE_SIZE = 1_000_000  # 1MB
MAX_WRITE_SIZE = 5_000_000  # 5MB
BASH_TIMEOUT = 30
BASH_OUTPUT_MAX = 100_000  # characters returned to LLM

DANGEROUS_COMMANDS = [
    "sudo", "su ", "chown", "chmod 777", "mkfs", "dd if=",
    "reboot", "shutdown", "init ", "halt", "poweroff",
]


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str | None = None
    truncated: bool = False


class ToolError(Exception):
    pass


class AskUserSignal(Exception):
    """Raised when agent calls ask_user — caught by engine to pause loop."""
    def __init__(self, tool_call_id: str, question: str):
        self.tool_call_id = tool_call_id
        self.question = question


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------

def resolve_workspace_path(workspace: Path, user_path: str) -> Path:
    if not user_path or user_path == ".":
        return workspace
    clean = user_path.lstrip("/\\")
    resolved = (workspace / clean).resolve()
    workspace_resolved = workspace.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise ToolError(f"Path traversal detected: {user_path}")
    return resolved


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool_use format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Path is relative to workspace root. Max 1MB.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create a new file or overwrite an existing file. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Make a surgical edit by finding and replacing text. If old_string appears multiple times, the tool will fail asking for more context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "old_string": {"type": "string", "description": "Text to find (must be unique)"},
                "new_string": {"type": "string", "description": "Text to replace with"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "bash",
        "description": "Execute a bash command in the workspace directory. Has a 30-second timeout. Use for build, test, git, and running code. Do NOT run interactive commands.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "glob",
        "description": "Search for files matching a glob pattern. Uses ** for recursion (e.g., '**/*.py').",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to search for"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search file contents for a regex pattern. Searches recursively from workspace root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "include": {"type": "string", "description": "Optional: file glob to filter (e.g., '*.py')"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "directory_list",
        "description": "List files and directories. Shows names, types, and sizes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path (default: '.')"},
            },
            "required": [],
        },
    },
    {
        "name": "ask_user",
        "description": "Ask the user a question when you need clarification, permission, or additional context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the internet for current information. Use this when you need up-to-date data, news, documentation, or any information outside the workspace. Always search in Chinese (中文) for best results with Chinese topics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "number", "description": "Maximum results (default: 5, max: 10)"},
            },
            "required": ["query"],
        },
    },
]

TOOL_NAME_MAP = {t["name"]: t for t in TOOL_DEFINITIONS}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def read_file(path: str, workspace: Path) -> ToolResult:
    fp = resolve_workspace_path(workspace, path)
    if not fp.exists():
        return ToolResult(success=False, error=f"File not found: {path}")
    if not fp.is_file():
        return ToolResult(success=False, error=f"Not a file: {path}")
    size = fp.stat().st_size
    if size > MAX_FILE_SIZE:
        return ToolResult(success=False, error=f"File too large ({size} bytes). Max {MAX_FILE_SIZE} bytes.")
    try:
        content = fp.read_text("utf-8")
    except (UnicodeDecodeError, PermissionError) as e:
        return ToolResult(success=False, error=f"Cannot read file: {e}")
    return ToolResult(success=True, output=content)


async def write_file(path: str, content: str, workspace: Path) -> ToolResult:
    fp = resolve_workspace_path(workspace, path)
    if len(content.encode("utf-8")) > MAX_WRITE_SIZE:
        return ToolResult(success=False, error=f"Content too large. Max {MAX_WRITE_SIZE} bytes.")
    fp.parent.mkdir(parents=True, exist_ok=True)
    try:
        fp.write_text(content, "utf-8")
    except PermissionError as e:
        return ToolResult(success=False, error=f"Cannot write file: {e}")
    return ToolResult(success=True, output=f"File written: {path}")


async def edit_file(path: str, old_string: str, new_string: str, workspace: Path) -> ToolResult:
    fp = resolve_workspace_path(workspace, path)
    if not fp.exists():
        return ToolResult(success=False, error=f"File not found: {path}")
    try:
        content = fp.read_text("utf-8")
    except (UnicodeDecodeError, PermissionError) as e:
        return ToolResult(success=False, error=f"Cannot read file: {e}")

    count = content.count(old_string)
    if count == 0:
        return ToolResult(success=False, error=f"old_string not found in {path}")
    if count > 1:
        return ToolResult(success=False, error=f"old_string found {count} times in {path}. Include more context to make it unique.")

    new_content = content.replace(old_string, new_string, 1)
    try:
        fp.write_text(new_content, "utf-8")
    except PermissionError as e:
        return ToolResult(success=False, error=f"Cannot write file: {e}")
    return ToolResult(success=True, output=f"File edited: {path}")


async def bash(command: str, workspace: Path, timeout: int = BASH_TIMEOUT) -> ToolResult:
    cmd_lower = command.lower()
    for pattern in DANGEROUS_COMMANDS:
        if pattern in cmd_lower:
            return ToolResult(success=False, error=f"Command blocked: contains '{pattern}'")

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workspace),
        env={
            **os.environ,
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return ToolResult(success=False, error=f"Command timed out after {timeout}s")

    output = stdout.decode("utf-8", errors="replace")
    if stderr:
        stderr_text = stderr.decode("utf-8", errors="replace")
        if stderr_text.strip():
            output += "\nSTDERR:\n" + stderr_text

    truncated = False
    if len(output) > BASH_OUTPUT_MAX:
        output = output[:BASH_OUTPUT_MAX] + f"\n... (truncated, {len(output)} total chars)"
        truncated = True

    return ToolResult(
        success=process.returncode == 0,
        output=output,
        truncated=truncated,
    )


async def glob_tool(pattern: str, workspace: Path) -> ToolResult:
    if not pattern.startswith("**") and not pattern.startswith("*"):
        pattern = "**/" + pattern
    try:
        matches = [str(p.relative_to(workspace)) for p in workspace.glob(pattern) if p.is_file()]
    except ValueError:
        matches = []
    if not matches:
        return ToolResult(success=True, output="No files found matching pattern.")
    result = "\n".join(sorted(matches))
    return ToolResult(success=True, output=result)


async def grep_tool(pattern: str, workspace: Path, include: str | None = None) -> ToolResult:
    matches = []
    for root, dirs, files in os.walk(workspace):
        root_path = Path(root)
        # Skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if include and not fn.endswith(include.replace("*", "")):
                continue
            fp = root_path / fn
            try:
                rel = fp.relative_to(workspace)
            except ValueError:
                continue
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            matches.append(f"{rel}:{i}:{line.rstrip()}")
            except PermissionError:
                continue
        if len(matches) > 200:
            break  # limit results

    if not matches:
        return ToolResult(success=True, output="No matches found.")
    result = "\n".join(matches[:200])
    if len(matches) > 200:
        result += f"\n... ({len(matches) - 200} more matches omitted)"
    return ToolResult(success=True, output=result)


async def directory_list(path: str, workspace: Path) -> ToolResult:
    fp = resolve_workspace_path(workspace, path) if path else workspace
    if not fp.exists():
        return ToolResult(success=False, error=f"Directory not found: {path}")
    if not fp.is_dir():
        return ToolResult(success=False, error=f"Not a directory: {path}")

    entries = sorted(fp.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = [f"📁 {p.name}/" if p.is_dir() else f"📄 {p.name} ({p.stat().st_size} B)" for p in entries]
    return ToolResult(success=True, output="\n".join(lines) if lines else "(empty)")


async def ask_user(tool_call_id: str, question: str) -> ToolResult:
    raise AskUserSignal(tool_call_id=tool_call_id, question=question)


async def web_search(query: str, workspace: Path, max_results: int = 5) -> ToolResult:
    """Search DuckDuckGo for current information. Free, no API key needed."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return ToolResult(success=False, error="Search library not installed. Run: pip install ddgs")

    max_results = min(max(max_results, 1), 10)

    def _search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _search)

        if not results:
            return ToolResult(success=True, output="No results found.")

        entries = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            entries.append(f"Title: {title}\nURL: {href}\nSnippet: {body}")

        output = f"Search results for \"{query}\":\n\n---\n\n" + "\n\n---\n\n".join(entries)
        return ToolResult(success=True, output=output)

    except Exception as e:
        error_msg = str(e)
        if "202" in error_msg or "ratelimit" in error_msg.lower() or "timeout" in error_msg.lower():
            return ToolResult(success=False, error="Search temporarily unavailable due to rate limiting. Try a different query or wait a moment.")
        return ToolResult(success=False, error=f"Search failed: {error_msg}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "bash": bash,
    "glob": glob_tool,
    "grep": grep_tool,
    "directory_list": directory_list,
    "ask_user": ask_user,
    "web_search": web_search,
}
