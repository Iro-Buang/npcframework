from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

JsonDict = Dict[str, Any]


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    # ISO-ish without timezone dependency; good enough for SQLite text sorting
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _json_dumps(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return "{}"


def _json_loads(s: Optional[str]) -> JsonDict:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"_": obj}
    except Exception:
        return {}


def _uuid() -> str:
    return uuid.uuid4().hex


@dataclass
class Event:
    # Keep your existing Event shape so the rest of your code doesn’t cry.
    id: int
    ts: float
    role: str         # "user" | "assistant" | "system"
    content: str
    meta: JsonDict


class NPCDatabase:
    """
    Drop-in replacement:
    - Preserves existing API: init_db, add_event, get_recent_events, set_state, get_state, get_state_many, wipe_events
    - Implements expanded schema (sessions/events/memories/relations/etc.)
    - Maintains integer event ids via event_seq for compatibility
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        npc_id: str = "npc",
        default_channel: str = "cli",
        default_environment_id: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.npc_id = npc_id
        self.default_channel = default_channel
        self.default_environment_id = default_environment_id

    # -------------------------
    # Connection / Setup
    # -------------------------

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            cur = conn.cursor()

            # ---- sessions
            cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at   TEXT,
                channel    TEXT,
                environment_id TEXT,
                meta_json  TEXT
            );
            """)

            # ---- events (new canonical)
            # event_seq keeps your old "id INTEGER AUTOINCREMENT" vibe
            cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_seq  INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id   TEXT UNIQUE NOT NULL,
                ts         REAL NOT NULL,
                ts_iso     TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),

                actor_type TEXT NOT NULL,   -- user | npc | system | environment | tool
                actor_id   TEXT,
                event_type TEXT NOT NULL,   -- message | tool_call | tool_result | state_change | observation
                content    TEXT,
                payload_json TEXT,

                importance_hint INTEGER NOT NULL DEFAULT 0,
                hash TEXT
            );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, ts);")

            # ---- state kv (keep)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS state_kv (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """)

            # ---- memories
            cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT,

                memory_type TEXT NOT NULL, -- episodic | semantic | entity | relation | preference | rule
                status TEXT NOT NULL DEFAULT 'active', -- active | archived | deleted

                title TEXT,
                summary TEXT,
                data_json TEXT,

                salience INTEGER NOT NULL DEFAULT 50,
                confidence REAL NOT NULL DEFAULT 0.7,

                valid_from TEXT,
                valid_to TEXT,

                scope TEXT NOT NULL DEFAULT 'npc', -- npc | shared | environment
                visibility TEXT NOT NULL DEFAULT 'private', -- private | public | user_visible
                pinned INTEGER NOT NULL DEFAULT 0
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_type_status ON memories(memory_type, status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience DESC);")

            # ---- memory_sources (provenance)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_sources (
                memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                event_id  TEXT NOT NULL, -- references events.event_id logically
                weight REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (memory_id, event_id)
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_sources_event ON memory_sources(event_id);")

            # ---- classifications
            cur.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                classification_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,

                target_type TEXT NOT NULL, -- event | memory
                target_id TEXT NOT NULL,

                classifier TEXT NOT NULL,  -- regex_v1 | keywords_v2 | manual | llm_v1
                label TEXT NOT NULL,
                score REAL NOT NULL,
                rationale TEXT,
                meta_json TEXT
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_classifications_target ON classifications(target_type, target_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_classifications_label ON classifications(label, score DESC);")

            # ---- entities
            cur.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,  -- person | place | org | object | concept
                name TEXT NOT NULL,
                aliases_json TEXT,
                attributes_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            );
            """)
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_type_name ON entities(entity_type, name);")

            # ---- relations
            cur.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                relation_id TEXT PRIMARY KEY,
                subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
                predicate TEXT NOT NULL,
                object_entity_id TEXT REFERENCES entities(entity_id) ON DELETE SET NULL,
                object_value TEXT,
                confidence REAL NOT NULL DEFAULT 0.7,
                salience INTEGER NOT NULL DEFAULT 50,
                updated_at TEXT NOT NULL
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_relations_subject_pred ON relations(subject_entity_id, predicate);")

            # ---- session summaries
            cur.execute("""
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                summary_style TEXT NOT NULL DEFAULT 'diary',  -- diary | bullet | forensic
                generated_by TEXT NOT NULL DEFAULT 'rules',   -- rules | llm
                updated_at TEXT NOT NULL
            );
            """)

            conn.commit()

    # -------------------------
    # Session helpers
    # -------------------------

    def _get_or_create_current_session(self, conn: sqlite3.Connection) -> str:
        """
        Simple session policy:
        - If state_kv['current_session_id'] exists and session not ended -> use it
        - Else create a new session and store it in state_kv
        """
        cur = conn.cursor()
        cur.execute("SELECT value_json FROM state_kv WHERE key = ?", ("current_session_id",))
        row = cur.fetchone()
        if row:
            sid = None
            try:
                sid = json.loads(row["value_json"])
            except Exception:
                sid = None
            if isinstance(sid, str) and sid:
                cur.execute("SELECT ended_at FROM sessions WHERE session_id = ?", (sid,))
                srow = cur.fetchone()
                if srow and (srow["ended_at"] is None):
                    return sid

        # create session
        session_id = f"sess_{_uuid()}"
        cur.execute(
            """INSERT INTO sessions (session_id, started_at, channel, environment_id, meta_json)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, _now_iso(), self.default_channel, self.default_environment_id, "{}"),
        )
        # store in state
        cur.execute("""
            INSERT INTO state_kv (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
        """, ("current_session_id", _json_dumps(session_id), _now_ts()))
        conn.commit()
        return session_id

    def end_session(self) -> None:
        """Optional: explicitly end the current session."""
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value_json FROM state_kv WHERE key = ?", ("current_session_id",))
            row = cur.fetchone()
            if not row:
                return
            try:
                sid = json.loads(row["value_json"])
            except Exception:
                return
            if not isinstance(sid, str) or not sid:
                return
            cur.execute("UPDATE sessions SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL", (_now_iso(), sid))
            conn.commit()

    # -------------------------
    # Events (Drop-in)
    # -------------------------

    def add_event(self, role: str, content: str, meta: Optional[JsonDict] = None) -> int:
        """
        Drop-in behavior:
        - role: "user" | "assistant" | "system" (original)
        - Stores to new canonical events table
        - Returns integer event_seq as "id" for compatibility
        """
        if meta is None:
            meta = {}
        ts = _now_ts()
        ts_iso = _now_iso()

        # Map old roles to new actor_type
        actor_type = role
        if role == "npc":
            actor_type = "assistant"
        elif role == "user":
            actor_type = "user"
        elif role == "system":
            actor_type = "system"
        else:
            actor_type = "unknown"

        # Basic event type inference; override via meta if you want
        event_type = str(meta.get("event_type") or "message")

        with self.connect() as conn:
            session_id = self._get_or_create_current_session(conn)
            cur = conn.cursor()

            event_id = f"evt_{_uuid()}"
            payload = dict(meta)
            # preserve compatibility metadata
            payload.setdefault("role", role)
            payload_json = _json_dumps(payload)

            cur.execute(
                """INSERT INTO events
                   (event_id, ts, ts_iso, session_id, actor_type, actor_id, event_type, content, payload_json, importance_hint, hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    ts,
                    ts_iso,
                    session_id,
                    actor_type,
                    payload.get("actor_id") or (self.npc_id if actor_type == "npc" else None),
                    event_type,
                    content,
                    payload_json,
                    int(payload.get("importance_hint", 0) or 0),
                    payload.get("hash"),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)  # event_seq

    def get_recent_events(self, limit: int = 20) -> List[Event]:
        """
        Drop-in behavior:
        - returns List[Event] in chronological order
        """
        with self.connect() as conn:
            cur = conn.cursor()
            # Most recent across ALL sessions; matches old behavior
            cur.execute(
                """SELECT event_seq, ts, actor_type, content, payload_json
                   FROM events
                   ORDER BY event_seq DESC
                   LIMIT ?""",
                (limit,),
            )
            rows = cur.fetchall()

        events: List[Event] = []
        for r in reversed(rows):
            payload = _json_loads(r["payload_json"])
            # Map back to old 'role'
            role = payload.get("role")
            if not role:
                # reverse-map actor_type
                at = str(r["actor_type"])
                role = "assistant" if at == "npc" else at

            events.append(
                Event(
                    id=int(r["event_seq"]),
                    ts=float(r["ts"]),
                    role=str(role),
                    content=str(r["content"] or ""),
                    meta=payload,
                )
            )
        return events

    def wipe_events(self) -> None:
        """
        Drop-in behavior:
        - Clears event history
        - Leaves derived tables intact (your choice)
        """
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM events;")
            # Optional: also clear sessions that are now empty
            cur.execute("DELETE FROM sessions;")
            conn.commit()

    # -------------------------
    # State (Drop-in)
    # -------------------------

    def set_state(self, key: str, value: Any) -> None:
        ts = _now_ts()
        value_json = _json_dumps(value)

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

    # -------------------------
    # New capabilities (optional but you’ll want them)
    # -------------------------

    def create_memory(
        self,
        memory_type: str,
        summary: str,
        title: Optional[str] = None,
        data: Optional[JsonDict] = None,
        salience: int = 50,
        confidence: float = 0.7,
        scope: str = "npc",
        visibility: str = "private",
        pinned: bool = False,
        source_event_seqs: Optional[List[int]] = None,
    ) -> str:
        """
        Create a derived memory item and optionally attach provenance from event_seq list.
        """
        memory_id = f"mem_{_uuid()}"
        created_at = _now_iso()
        data_json = _json_dumps(data or {})

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO memories
                   (memory_id, created_at, updated_at, memory_type, status, title, summary, data_json,
                    salience, confidence, scope, visibility, pinned)
                   VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory_id, created_at, created_at, memory_type,
                    title, summary, data_json,
                    int(salience), float(confidence),
                    scope, visibility, 1 if pinned else 0
                ),
            )

            if source_event_seqs:
                # translate event_seq -> event_id for stable linking
                q_marks = ",".join(["?"] * len(source_event_seqs))
                cur.execute(
                    f"SELECT event_id FROM events WHERE event_seq IN ({q_marks})",
                    tuple(source_event_seqs),
                )
                eids = [row["event_id"] for row in cur.fetchall()]
                for eid in eids:
                    cur.execute(
                        "INSERT OR IGNORE INTO memory_sources (memory_id, event_id, weight) VALUES (?, ?, 1.0)",
                        (memory_id, eid),
                    )

            conn.commit()

        return memory_id

    def add_classification(
        self,
        target_type: str,  # 'event' | 'memory'
        target_id: str,
        classifier: str,
        label: str,
        score: float,
        rationale: Optional[str] = None,
        meta: Optional[JsonDict] = None,
    ) -> str:
        cid = f"cls_{_uuid()}"
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO classifications
                   (classification_id, created_at, target_type, target_id, classifier, label, score, rationale, meta_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cid,
                    _now_iso(),
                    target_type,
                    target_id,
                    classifier,
                    label,
                    float(score),
                    rationale,
                    _json_dumps(meta or {}),
                ),
            )
            conn.commit()
        return cid
