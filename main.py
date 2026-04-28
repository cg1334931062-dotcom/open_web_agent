import asyncio
import json
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from agent.engine import Agent, AgentSettings
from session_store import SessionStore


HERE = Path(__file__).parent
STATIC_DIR = HERE / "static"
INDEX_HTML = STATIC_DIR / "index.html"
DATA_DIR = HERE / "data"


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    agent: Agent
    pending_ask_id: str | None = None
    canceled: bool = False
    message_checkpoint: int = 0  # index in agent.messages before current turn
    had_ask_user: bool = False   # skip suggestions if turn involved ask_user
    _llm_error: bool = False     # skip suggestions after LLM error


class AppState:
    def __init__(self):
        self.sessions: dict[str, SessionState] = {}
        self.settings = AgentSettings(
            skill_dirs=[str(HERE / "skills")],
        )
        self.store = SessionStore(str(DATA_DIR / "web-agent.db"))
        self._store_ready = False

    async def ensure_store(self):
        """Lazily initialise the database on first use."""
        if not self._store_ready:
            await self.store.init()
            self._store_ready = True

    async def create_session(self, workspace_override: str = "") -> tuple[str, SessionState]:
        sid = uuid.uuid4().hex[:12]
        s = self.settings
        wsp = workspace_override or s.workspace_path or str(Path.cwd())
        settings = AgentSettings(
            provider=s.provider, api_key=s.api_key, model=s.model,
            base_url=s.base_url,
            workspace_path=wsp,
            skill_dirs=s.skill_dirs,
        )
        agent = Agent(settings, store=self.store)
        agent.initialize()
        session = SessionState(agent=agent)
        self.sessions[sid] = session
        return sid, session

    def create_session_sync(self) -> tuple[str, SessionState]:
        """Synchronous wrapper for legacy callers. Returns immediately."""
        sid = uuid.uuid4().hex[:12]
        s = self.settings
        settings = AgentSettings(
            provider=s.provider, api_key=s.api_key, model=s.model,
            base_url=s.base_url,
            workspace_path=s.workspace_path or str(Path.cwd()),
            skill_dirs=s.skill_dirs,
        )
        agent = Agent(settings, store=self.store)
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
# Session history API
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions():
    await state.ensure_store()
    sessions = await state.store.list_sessions()
    return JSONResponse({"sessions": sessions})


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    await state.ensure_store()
    deleted = await state.store.delete_session(sid)
    if not deleted:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    state.sessions.pop(sid, None)
    return JSONResponse({"ok": True})


@app.get("/api/memories")
async def list_memories():
    await state.ensure_store()
    memories = await state.store.memory_list()
    return JSONResponse({"memories": memories})


@app.delete("/api/memories/{mid}")
async def delete_memory(mid: int):
    await state.ensure_store()
    deleted = await state.store.memory_delete(mid)
    if not deleted:
        return JSONResponse({"error": "Memory not found"}, status_code=404)
    return JSONResponse({"ok": True})


@app.get("/api/sessions/{sid}/messages")
async def get_session_messages(sid: str):
    await state.ensure_store()
    record = await state.store.get_session(sid)
    if not record:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    messages = await state.store.load_messages(sid)
    return JSONResponse({"session": record, "messages": messages})


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Ensure DB is ready, then check for session resume
    await state.ensure_store()
    resume_sid = ws.query_params.get("session_id")
    resume_messages: list | None = None

    if resume_sid:
        record = await state.store.get_session(resume_sid)
        if record:
            wsp = record.get("workspace", "")
            sid, session = await state.create_session(workspace_override=wsp)
            agent = session.agent
            resume_messages = await state.store.load_messages(resume_sid)
            agent.messages = resume_messages
            agent.tasks = await state.store.load_tasks(resume_sid)
            # Re-init agent with stored provider/model
            agent.settings.provider = record.get("provider", "anthropic")
            agent.settings.model = record.get("model", "")
            agent.settings.base_url = record.get("base_url", "")
            agent.initialize()
            # Remap to original session ID so DB writes target the right record
            state.sessions.pop(sid, None)
            sid = resume_sid
            state.sessions[sid] = session
        else:
            sid, session = await state.create_session()
            agent = session.agent
    else:
        sid, session = await state.create_session()
        agent = session.agent

    async def send(event: dict):
        try:
            await ws.send_json(event)
        except WebSocketDisconnect:
            raise

    try:
        await send({"type": "session_created", "session_id": sid})
        await send({"type": "info", "message": f"Connected. Workspace: {agent.workspace}"})

        # Send history and tasks if resuming a session
        if resume_messages:
            await send({"type": "history", "messages": resume_messages})
        # Always restore tasks when resuming
        if agent.tasks:
            max_id = max((int(t["id"]) for t in agent.tasks if t["id"].isdigit()), default=0)
            agent._task_counter = max_id
        await send({"type": "task_update", "tasks": agent.tasks})

        agent_task: asyncio.Task | None = None

        async def stop_agent():
            """Cancel current agent task by setting signal and waiting."""
            nonlocal agent_task
            if agent_task and not agent_task.done():
                session.canceled = True
                try:
                    await asyncio.wait_for(agent_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    agent_task.cancel()
                agent_task = None

        async def run_agent_in_background():
            """Run agent turn, persist messages, and send turn_complete."""
            try:
                await run_agent_turn(agent, send, session)
                # Skip turn_complete if paused for ask_user (will be resumed)
                if session.pending_ask_id:
                    return
                if not session.canceled and not session._llm_error:
                    suggestions = []
                    if not session.had_ask_user:
                        try:
                            suggestions = await agent.generate_suggestions()
                        except Exception:
                            pass
                    session.had_ask_user = False
                    await send({"type": "turn_complete", "suggestions": suggestions})
                    # Persist messages after turn completes
                    asyncio.create_task(_persist_session())
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Agent task error: {e}")
                traceback.print_exc()

        async def _persist_session():
            """Fire-and-forget: save messages, tasks, and update session timestamp."""
            try:
                await state.store.save_messages(sid, agent.messages)
                await state.store.save_tasks(sid, agent.tasks)
                await state.store.update_session(sid)
            except Exception as e:
                print(f"DB persist error: {e}")

        # Fire-and-forget: create session record in DB and clean up old blanks
        s_cfg = agent.settings
        asyncio.create_task(
            state.store.create_session(
                sid=sid,
                workspace=str(agent.workspace),
                provider=s_cfg.provider,
                model=s_cfg.model,
                base_url=s_cfg.base_url or "",
            )
        )
        asyncio.create_task(state.store.cleanup_blank_sessions(sid))

        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "")

            # --- Settings ---
            if msg_type == "settings_update":
                await stop_agent()
                s = data.get("settings", {})
                old_key = state.settings.api_key
                providers_raw = s.get("providers")
                active = s.get("active_provider", "")
                state.settings = AgentSettings(
                    provider=s.get("provider", "anthropic"),
                    api_key=s.get("apiKey") or s.get("api_key") or old_key,
                    model=s.get("model", ""),
                    base_url=s.get("baseUrl", ""),
                    workspace_path=s.get("workspace") or s.get("workspace_path") or state.settings.workspace_path or "",
                    skill_dirs=s.get("skill_dirs", state.settings.skill_dirs),
                    providers=providers_raw if isinstance(providers_raw, list) else None,
                    active_provider=active,
                )
                old_messages = agent.messages
                new_agent = Agent(state.settings, store=state.store)
                new_agent.initialize()
                new_agent.messages = old_messages
                new_agent.tasks = agent.tasks
                new_agent._task_counter = agent._task_counter
                session.agent = new_agent
                agent = new_agent
                await send({"type": "info", "message": "Settings updated"})
                await send({"type": "skills_updated", "skills": agent.skills.get_skills_info()})
                await send({"type": "providers_updated", "providers": state.settings.get_provider_list(), "active_provider": state.settings.active_provider})
                continue

            # --- Toggle skill ---
            if msg_type == "toggle_skill":
                ok = agent.skills.toggle_skill(data["name"], data.get("enabled", True))
                await send({"type": "skills_updated", "skills": agent.skills.get_skills_info()})
                continue

            if msg_type == "task_toggle":
                tid = data.get("id", "")
                task = next((t for t in agent.tasks if t["id"] == tid), None)
                if task:
                    task["status"] = "completed" if task["status"] != "completed" else "pending"
                    await send({"type": "task_update", "tasks": agent.tasks})
                    asyncio.create_task(state.store.save_tasks(sid, agent.tasks))
                continue

            # --- Ask user response ---
            if msg_type == "ask_user_response":
                if session.pending_ask_id:
                    agent.inject_ask_user_response(
                        response=data.get("text", ""),
                        tool_call_id=session.pending_ask_id,
                    )
                    session.pending_ask_id = None
                    session.canceled = False
                    agent_task = asyncio.create_task(run_agent_in_background())
                continue

            # --- Abort ---
            if msg_type == "abort":
                await stop_agent()
                # Roll back conversation: remove the aborted message and any
                # partial tool results / assistant responses that were added
                # after the checkpoint.
                while len(agent.messages) > session.message_checkpoint:
                    agent.messages.pop()
                continue

            # --- User message ---
            if msg_type == "message":
                text = data.get("text", "").strip()
                if not text:
                    continue
                await stop_agent()
                session.canceled = False
                session.message_checkpoint = len(agent.messages)
                agent.messages.append({"role": "user", "content": text})
                await send({"type": "user_message", "text": text})

                # Auto-title: first user message becomes the session title
                if len(agent.messages) == 1:
                    title = text[:100].replace("\n", " ").strip()
                    asyncio.create_task(state.store.update_session(sid, title=title))

                agent_task = asyncio.create_task(run_agent_in_background())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        tb = traceback.format_exc()
        print(f"WebSocket error: {e}\n{tb}")
        try:
            await send({"type": "error", "message": f"Server error: {str(e)}"})
        except Exception:
            pass
    finally:
        # Persist final state on disconnect, then clean up in-memory
        try:
            asyncio.create_task(state.store.save_messages(sid, agent.messages))
            asyncio.create_task(state.store.save_tasks(sid, agent.tasks))
        except Exception:
            pass
        state.sessions.pop(sid, None)


# ---------------------------------------------------------------------------
# Agent turn loop
# ---------------------------------------------------------------------------

async def run_agent_turn(agent: Agent, send, session: SessionState):
    """Run one agent turn: LLM call → tool execution → repeat until done."""
    session._llm_error = False
    for _ in range(25):  # max 25 turns per user message
        if session.canceled:
            return

        # --- Step 1: Flush any pending tool results, then call LLM ---
        agent.flush_tool_results()
        await send({"type": "thinking_start"})

        has_content = False
        async for event in agent.stream_llm_call():
            if session.canceled:
                await send({"type": "thinking_end"})
                return
            etype = event["type"]
            if etype == "thinking_delta":
                await send(event)
            elif etype == "text_delta":
                has_content = True
                await send(event)
            elif etype == "tool_call_start":
                has_content = True
                await send(event)
            elif etype == "text_end":
                await send(event)
            elif etype == "error":
                err_msg = event.get('message', '').strip() or 'LLM call failed'
                session._llm_error = True
                await send({"type": "error", "message": f"LLM error: {err_msg}"})
                # Reset any in_progress task back to pending
                for t in agent.tasks:
                    if t["status"] == "in_progress":
                        t["status"] = "pending"
                await send({"type": "task_update", "tasks": agent.tasks})
                return

        if session.canceled:
            return

        # Always send thinking_end to clear the UI state
        await send({"type": "thinking_end"})

        # If LLM returned no text and no tool calls, notify user
        if not has_content and not agent.has_tool_uses:
            await send({"type": "error", "message": "AI returned empty response. Check API key and network."})
            return

        # Commit response to conversation history
        agent.commit_llm_response()

        # If no tool calls, done
        if not agent.has_tool_uses:
            return

        # --- Step 2: Execute each tool ---
        for tu in agent.pending_tool_uses:
            if session.canceled:
                return
            tid, name, args = tu["id"], tu["name"], tu["input"]

            async for event in agent.execute_tool(tid, name, args):
                if event["type"] == "ask_user":
                    # Pause — wait for user response
                    session.pending_ask_id = tid
                    session.had_ask_user = True
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
