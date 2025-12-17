from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

JsonDict = Dict[str, Any]


@dataclass
class Event:
    id: int
    ts: float
    role: str         # "user" | "assistant" | "system"
    content: str
    meta: JsonDict


class NPCDatabase:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            cur = conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}'
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS state_kv (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """)

            conn.commit()

    # -------------------------
    # Events
    # -------------------------

    def add_event(self, role: str, content: str, meta: Optional[JsonDict] = None) -> int:
        if meta is None:
            meta = {}
        ts = time.time()
        meta_json = json.dumps(meta, ensure_ascii=False)

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO event_log (ts, role, content, meta_json) VALUES (?, ?, ?, ?)",
                (ts, role, content, meta_json),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_recent_events(self, limit: int = 20) -> List[Event]:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, ts, role, content, meta_json FROM event_log ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()

        # reverse to chronological
        events: List[Event] = []
        for r in reversed(rows):
            try:
                meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
            except Exception:
                meta = {}
            events.append(
                Event(
                    id=int(r["id"]),
                    ts=float(r["ts"]),
                    role=str(r["role"]),
                    content=str(r["content"]),
                    meta=meta,
                )
            )
        return events

    # -------------------------
    # State
    # -------------------------

    def set_state(self, key: str, value: Any) -> None:
        ts = time.time()
        value_json = json.dumps(value, ensure_ascii=False)

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO state_kv (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
            """, (key, value_json, ts))
            conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value_json FROM state_kv WHERE key = ?", (key,))
            row = cur.fetchone()

        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except Exception:
            return default

    def get_state_many(self, keys: list[str]) -> dict:
        return {k: self.get_state(k) for k in keys}

    def wipe_events(self) -> None:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM event_log;")
            conn.commit()
