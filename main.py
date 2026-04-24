import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from agent.engine import Agent, AgentSettings


HERE = Path(__file__).parent
STATIC_DIR = HERE / "static"
INDEX_HTML = STATIC_DIR / "index.html"


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    agent: Agent
    pending_ask_id: str | None = None
    canceled: bool = False


class AppState:
    def __init__(self):
        self.sessions: dict[str, SessionState] = {}
        self.settings = AgentSettings(
            skill_dirs=[str(HERE / "skills")],
        )

    def create_session(self) -> tuple[str, SessionState]:
        sid = uuid.uuid4().hex[:12]
        s = self.settings
        settings = AgentSettings(
            provider=s.provider, api_key=s.api_key, model=s.model,
            base_url=s.base_url,
            workspace_path=s.workspace_path or str(Path.cwd()),
            skill_dirs=s.skill_dirs,
        )
        agent = Agent(settings)
        agent.initialize()
        session = SessionState(agent=agent)
        self.sessions[sid] = session
        return sid, session


state = AppState()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

app = FastAPI()


@app.get("/")
async def index():
    if not INDEX_HTML.exists():
        return JSONResponse({"error": "Frontend not found"}, status_code=500)
    return FileResponse(str(INDEX_HTML))


@app.get("/api/workspace/tree")
async def workspace_tree(workspace: str = ""):
    raw = workspace or state.settings.workspace_path or ""
    ws = Path(raw).resolve()
    if not ws.exists():
        return JSONResponse({"error": "Workspace path does not exist"}, status_code=400)

    def build_tree(path: Path, depth: int = 4):
        if depth <= 0:
            return {"name": path.name, "type": "directory", "children": []}
        entries = []
        try:
            for p in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if p.name.startswith("."):
                    continue
                if p.is_dir():
                    entries.append(build_tree(p, depth - 1))
                else:
                    try:
                        size = p.stat().st_size
                    except OSError:
                        size = 0
                    entries.append({"name": p.name, "type": "file", "size": size})
        except PermissionError:
            pass
        return {"name": path.name, "type": "directory", "children": entries}

    return JSONResponse({"tree": build_tree(ws), "root": str(ws)})


@app.get("/api/workspace/file")
async def workspace_file(path: str):
    ws = Path(state.settings.workspace_path or Path.cwd())
    fp = (ws / path.lstrip("/\\")).resolve()
    try:
        fp.relative_to(ws.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    if not fp.exists() or not fp.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)
    if fp.stat().st_size > 1_000_000:
        return JSONResponse({"error": "File too large"}, status_code=413)
    try:
        content = fp.read_text("utf-8")
    except (UnicodeDecodeError, PermissionError):
        try:
            content = fp.read_bytes().hex()[:200]
            content = f"[Binary file]\n{content}"
        except Exception:
            content = "[Cannot preview]"
    return JSONResponse({"content": content, "name": fp.name})


@app.get("/api/workspace/browse")
async def workspace_browse(path: str = ""):
    """List directories on the server for the directory picker."""
    if not path or path == "/":
        # Start from root or home
        dirs = [{"name": "/", "path": "/"}, {"name": "~", "path": str(Path.home())}]
        return JSONResponse({"dirs": dirs, "current": ""})

    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return JSONResponse({"error": "Directory not found"}, status_code=400)

    parent = str(p.parent)
    entries = []
    try:
        for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and not entry.is_symlink():
                try:
                    entries.append({"name": entry.name, "path": str(entry.resolve())})
                except Exception:
                    pass
    except PermissionError:
        pass

    return JSONResponse({"dirs": entries, "current": str(p), "parent": parent})


@app.get("/api/workspace/pick-folder")
async def pick_folder():
    """Open a native OS folder picker dialog.
    Uses osascript (macOS) or zenity/kdialog (Linux). Falls back to tkinter."""
    import sys
    import shutil

    # macOS: use osascript for native dialog
    if sys.platform == "darwin" and shutil.which("osascript"):
        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e",
                'POSIX path of (choose folder with prompt "Select workspace folder for Web Agent")',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                path = stdout.decode("utf-8").strip().rstrip("/")
                if path:
                    return JSONResponse({"path": path})
            return JSONResponse({"path": None})
        except Exception:
            pass  # fall through to browser-based picker

    # Linux: try zenity (GNOME) or kdialog (KDE)
    dialog_cmd = None
    if shutil.which("zenity"):
        dialog_cmd = ["zenity", "--file-selection", "--directory", "--title=Select workspace folder"]
    elif shutil.which("kdialog"):
        dialog_cmd = ["kdialog", "--getexistingdirectory", "--title=Select workspace folder"]

    if dialog_cmd:
        try:
            proc = await asyncio.create_subprocess_exec(
                *dialog_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                path = stdout.decode("utf-8").strip().rstrip("/")
                if path:
                    return JSONResponse({"path": path})
            return JSONResponse({"path": None})
        except Exception:
            pass  # fall through

    # Windows: try PowerShell folder browser
    if sys.platform == "win32":
        try:
            ps_script = '''
Add-Type -AssemblyName System.Windows.Forms
$f = New-Object System.Windows.Forms.FolderBrowserDialog
$f.Description = "Select workspace folder"
$f.ShowNewFolderButton = $false
if ($f.ShowDialog() -eq "OK") { Write-Output $f.SelectedPath }
'''
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command", ps_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                path = stdout.decode("utf-8").strip().rstrip("\r\n")
                if path:
                    return JSONResponse({"path": path})
            return JSONResponse({"path": None})
        except Exception:
            pass  # fall through

    return JSONResponse({"path": None, "error": "No native dialog available. Use the browser-based picker."})


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    sid, session = state.create_session()
    agent = session.agent

    async def send(event: dict):
        try:
            await ws.send_json(event)
        except WebSocketDisconnect:
            raise

    try:
        await send({"type": "session_created", "session_id": sid})
        await send({"type": "info", "message": f"Connected. Workspace: {agent.workspace}"})

        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "")

            # --- Settings ---
            if msg_type == "settings_update":
                s = data.get("settings", {})
                old_key = state.settings.api_key
                # Map frontend camelCase keys to backend snake_case
                state.settings = AgentSettings(
                    provider=s.get("provider", "anthropic"),
                    api_key=s.get("apiKey") or s.get("api_key") or old_key,
                    model=s.get("model", ""),
                    base_url=s.get("baseUrl", ""),
                    workspace_path=s.get("workspace") or s.get("workspace_path") or state.settings.workspace_path or "",
                    skill_dirs=s.get("skill_dirs", state.settings.skill_dirs),
                )
                new_agent = Agent(state.settings)
                new_agent.initialize()
                session.agent = new_agent
                agent = new_agent
                await send({"type": "info", "message": "Settings updated"})
                await send({"type": "skills_updated", "skills": agent.skills.get_skills_info()})
                continue

            # --- Toggle skill ---
            if msg_type == "toggle_skill":
                ok = agent.skills.toggle_skill(data["name"], data.get("enabled", True))
                await send({"type": "skills_updated", "skills": agent.skills.get_skills_info()})
                continue

            # --- Ask user response ---
            if msg_type == "ask_user_response":
                if session.pending_ask_id:
                    agent.inject_ask_user_response(
                        response=data.get("text", ""),
                        tool_call_id=session.pending_ask_id,
                    )
                    session.pending_ask_id = None
                    # Resume: call LLM again with the injected response
                    await run_agent_turn(agent, send, session)
                    await send({"type": "turn_complete"})
                continue

            # --- User message ---
            if msg_type == "message":
                text = data.get("text", "").strip()
                if not text:
                    continue
                session.canceled = False
                agent.messages.append({"role": "user", "content": text})
                await send({"type": "user_message", "text": text})
                await run_agent_turn(agent, send, session)
                await send({"type": "turn_complete"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await send({"type": "error", "message": f"Server error: {str(e)}"})
        except Exception:
            pass
    finally:
        state.sessions.pop(sid, None)


# ---------------------------------------------------------------------------
# Agent turn loop
# ---------------------------------------------------------------------------

async def run_agent_turn(agent: Agent, send, session: SessionState):
    """Run one agent turn: LLM call → tool execution → repeat until done."""
    for _ in range(25):  # max 25 turns per user message
        if session.canceled:
            return

        # --- Step 1: Flush any pending tool results, then call LLM ---
        agent.flush_tool_results()
        await send({"type": "thinking_start"})

        async for event in agent.stream_llm_call():
            if event["type"] == "thinking_delta":
                await send(event)
            elif event["type"] == "text_delta":
                await send(event)
            elif event["type"] == "tool_call_start":
                await send(event)
            elif event["type"] == "text_end":
                await send(event)
            elif event["type"] == "error":
                await send(event)
                return

        # Always send thinking_end to clear the UI state
        await send({"type": "thinking_end"})

        # Commit response to conversation history
        agent.commit_llm_response()

        # If no tool calls, done
        if not agent.has_tool_uses:
            return

        # --- Step 2: Execute each tool ---
        for tu in agent.pending_tool_uses:
            tid, name, args = tu["id"], tu["name"], tu["input"]

            async for event in agent.execute_tool(tid, name, args):
                if event["type"] == "ask_user":
                    # Pause — wait for user response
                    session.pending_ask_id = tid
                    await send(event)
                    return  # will be resumed via ask_user_response
                else:
                    await send(event)

        # Flush all tool results as a single user message (Anthropic requirement)
        agent.flush_tool_results()

        # --- Step 3: loop back to call LLM again with tool results ---

    await send({"type": "error", "message": "Reached max 25 turns. Simplify your request."})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
