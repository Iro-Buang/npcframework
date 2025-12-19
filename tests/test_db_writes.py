from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, List


class NPCDatabase:
    """
    SQLite-backed event/state store.

    Public contract:
    - .path (str): filesystem path to sqlite db file
    - .db_path (Path): pathlib view of path
    """

    def __init__(self, db_path: str | Path) -> None:
        p = Path(db_path)

        # Make sure parent dir exists (common in temp tests / fresh npc dirs)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Public + stable (tests and callers can rely on this)
        self.path: str = str(p)
        self._db_path: Path = p

    @property
    def db_path(self) -> Path:
        return self._db_path

    # --- your existing methods below ---
    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def init_db(self) -> None:
        con = self.connect()
        cur = con.cursor()

        # Keep whatever your schema actually is; this is a safe default
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT (datetime('now')),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                meta_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS state_kv (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        con.commit()
        con.close()

    def add_event(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        import json
        con = self.connect()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO event_log (role, content, meta_json) VALUES (?, ?, ?)",
            (role, content, json.dumps(meta) if meta else None),
        )
        con.commit()
        con.close()

    def get_state(self, key: str, default: Any = None) -> Any:
        con = self.connect()
        cur = con.cursor()
        cur.execute("SELECT value FROM state_kv WHERE key = ?", (key,))
        row = cur.fetchone()
        con.close()
        return row[0] if row else default

    def set_state(self, key: str, value: Any) -> None:
        con = self.connect()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO state_kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        con.commit()
        con.close()

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, str]]:
        con = self.connect()
        cur = con.cursor()
        cur.execute(
            "SELECT role, content FROM event_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        con.close()

        # Return chronological order
        rows.reverse()
        return [{"role": r, "content": c} for (r, c) in rows]
