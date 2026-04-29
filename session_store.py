"""
SQLite-backed session and message persistence.

Uses sqlite3 via asyncio.to_thread() to avoid blocking the event loop.
No extra dependencies — sqlite3 and asyncio are in the Python 3.11+ stdlib.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    workspace TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT 'anthropic',
    model TEXT NOT NULL DEFAULT '',
    base_url TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, sequence);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    sequence INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tasks_session
    ON tasks(session_id);

CREATE TABLE IF NOT EXISTS daily_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_daily_usage_date ON daily_usage(date);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Create tables and enable WAL mode. Idempotent."""
        if self._ready:
            return

        def _init():
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)
            conn.commit()
            conn.close()

        await asyncio.to_thread(_init)
        self._ready = True

    def _ensure_ready(self):
        if not self._ready:
            raise RuntimeError("SessionStore not initialised — call await store.init() first")

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    async def create_session(
        self,
        sid: str,
        title: str = "",
        workspace: str = "",
        provider: str = "",
        model: str = "",
        base_url: str = "",
    ) -> None:
        self._ensure_ready()
        now = _now()

        def _create():
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, title, workspace, provider, model, base_url, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sid, title, workspace, provider, model, base_url, now, now),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_create)

    async def update_session(self, sid: str, **fields) -> None:
        self._ensure_ready()
        allowed = {"title", "workspace", "provider", "model", "base_url", "updated_at"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = _now()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [sid]

        def _update():
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"UPDATE sessions SET {set_clause} WHERE id=?", values)
            conn.commit()
            conn.close()

        await asyncio.to_thread(_update)

    async def list_sessions(self) -> list[dict]:
        self._ensure_ready()

        def _list():
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT s.*, COUNT(m.id) AS message_count
                   FROM sessions s
                   LEFT JOIN messages m ON m.session_id = s.id
                   GROUP BY s.id
                   ORDER BY s.updated_at DESC"""
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

        return await asyncio.to_thread(_list)

    async def get_session(self, sid: str) -> dict | None:
        self._ensure_ready()

        def _get():
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
            conn.close()
            return dict(row) if row else None

        return await asyncio.to_thread(_get)

    async def delete_session(self, sid: str) -> bool:
        self._ensure_ready()

        def _delete():
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            cur = conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
            conn.commit()
            deleted = cur.rowcount > 0
            conn.close()
            return deleted

        return await asyncio.to_thread(_delete)

    # ------------------------------------------------------------------
    # Message persistence
    # ------------------------------------------------------------------

    async def save_messages(self, sid: str, messages: list) -> None:
        """Atomically replace all messages for a session.

        Called after each agent turn completes. Deletes existing messages
        and re-inserts the full list inside a single transaction.
        """
        self._ensure_ready()
        now = _now()

        def _save():
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN")
            try:
                conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
                rows = []
                for seq, msg in enumerate(messages):
                    content = msg.get("content", "")
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    rows.append((sid, msg.get("role", "user"), content, seq, now))
                conn.executemany(
                    "INSERT INTO messages (session_id, role, content, sequence, created_at) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                conn.execute(
                    "UPDATE sessions SET updated_at=? WHERE id=?", (now, sid)
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        await asyncio.to_thread(_save)

    async def load_messages(self, sid: str) -> list[dict]:
        """Load messages ordered by sequence, with content JSON-deserialised."""
        self._ensure_ready()

        def _load():
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT role, content, sequence FROM messages WHERE session_id=? ORDER BY sequence",
                (sid,),
            ).fetchall()
            conn.close()
            messages = []
            for row in rows:
                content = row["content"]
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass
                messages.append({"role": row["role"], "content": content})
            return messages

        return await asyncio.to_thread(_load)

    async def cleanup_blank_sessions(self, keep_sid: str) -> None:
        """Delete blank sessions (0 messages) except the one being kept."""
        self._ensure_ready()

        def _cleanup():
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                """DELETE FROM sessions WHERE id != ? AND id IN
                   (SELECT s.id FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
                    GROUP BY s.id HAVING COUNT(m.id) = 0)""",
                (keep_sid,),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_cleanup)

    # ------------------------------------------------------------------
    # Task persistence
    # ------------------------------------------------------------------

    async def save_tasks(self, sid: str, tasks: list) -> None:
        self._ensure_ready()

        def _save():
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN")
            try:
                conn.execute("DELETE FROM tasks WHERE session_id=?", (sid,))
                for i, t in enumerate(tasks):
                    conn.execute(
                        "INSERT INTO tasks (session_id, task_id, title, description, status, sequence) VALUES (?, ?, ?, ?, ?, ?)",
                        (sid, t["id"], t.get("title", ""), t.get("description", ""), t.get("status", "pending"), i),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        await asyncio.to_thread(_save)

    async def load_tasks(self, sid: str) -> list[dict]:
        self._ensure_ready()

        def _load():
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT task_id, title, description, status FROM tasks WHERE session_id=? ORDER BY sequence",
                (sid,),
            ).fetchall()
            conn.close()
            return [{"id": r["task_id"], "title": r["title"], "description": r["description"], "status": r["status"]} for r in rows]

        return await asyncio.to_thread(_load)

    # ------------------------------------------------------------------
    # Daily token usage
    # ------------------------------------------------------------------

    async def add_usage(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
        self._ensure_ready()
        today = datetime.now().strftime("%Y-%m-%d")

        def _add():
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute(
                "SELECT id, input_tokens, output_tokens FROM daily_usage WHERE date=? AND provider=? AND model=?",
                (today, provider, model),
            )
            row = cur.fetchone()
            if row:
                conn.execute(
                    "UPDATE daily_usage SET input_tokens=input_tokens+?, output_tokens=output_tokens+? WHERE id=?",
                    (input_tokens, output_tokens, row[0]),
                )
            else:
                conn.execute(
                    "INSERT INTO daily_usage (date, provider, model, input_tokens, output_tokens) VALUES (?,?,?,?,?)",
                    (today, provider, model, input_tokens, output_tokens),
                )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_add)

    async def get_daily_usage(self, days: int = 30) -> list[dict]:
        self._ensure_ready()
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        def _get():
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT date, SUM(input_tokens) as input_tokens, SUM(output_tokens) as output_tokens FROM daily_usage WHERE date >= ? GROUP BY date ORDER BY date DESC",
                (from_date,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

        return await asyncio.to_thread(_get)

    # ------------------------------------------------------------------
    # Memory persistence (global, not per-session)
    # ------------------------------------------------------------------

    async def memory_create(self, title: str, content: str) -> int:
        self._ensure_ready()
        now = _now()

        def _create():
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute(
                "INSERT INTO memories (title, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (title, content, now, now),
            )
            conn.commit()
            rowid = cur.lastrowid
            conn.close()
            return rowid

        return await asyncio.to_thread(_create)

    async def memory_search(self, query: str) -> list[dict]:
        self._ensure_ready()

        def _search():
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, title, content, created_at, updated_at FROM memories WHERE title LIKE ? OR content LIKE ? ORDER BY updated_at DESC LIMIT 20",
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

        return await asyncio.to_thread(_search)

    async def memory_list(self) -> list[dict]:
        self._ensure_ready()

        def _list():
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, title, content, created_at, updated_at FROM memories ORDER BY updated_at DESC"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

        return await asyncio.to_thread(_list)

    async def memory_delete(self, mid: int) -> bool:
        self._ensure_ready()

        def _delete():
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute("DELETE FROM memories WHERE id=?", (mid,))
            conn.commit()
            deleted = cur.rowcount > 0
            conn.close()
            return deleted

        return await asyncio.to_thread(_delete)
