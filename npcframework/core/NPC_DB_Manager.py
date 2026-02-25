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
    - Expanded schema (sessions/events/memories/relations/etc.)
    - Adds richer event provenance (channel/environment/correlation/parent/status)
    - Auto-migrates older DBs by adding missing columns/indexes
    - Maintains integer event ids via event_seq for compatibility
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        npc_id: str = "npc",
        default_channel: str = "cli",
        default_environment_id: Optional[str] = None,
        *,
        run_across_sessions: bool = False,
        run_across_environments: bool = False,
        run_across_channels: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.npc_id = npc_id
        self.default_channel = default_channel
        self.default_environment_id = default_environment_id
        self.run_across_sessions = bool(run_across_sessions)
        self.run_across_environments = bool(run_across_environments)
        self.run_across_channels = bool(run_across_channels)
        self._session_id_override = session_id

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

            # ---- events
            cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_seq  INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id   TEXT UNIQUE NOT NULL,
                ts         REAL NOT NULL,
                ts_iso     TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),

                channel        TEXT,
                environment_id TEXT,

                actor_type TEXT NOT NULL,   -- user | assistant | system | environment | tool
                actor_id   TEXT,

                event_type TEXT NOT NULL,   -- message | tool_call | tool_result | state_change | observation
                modality   TEXT,            -- text | image | audio | video | structured
                content_format TEXT,        -- plain | markdown | json | code | html | yaml

                content    TEXT,
                payload_json TEXT,

                parent_event_id TEXT,       -- causal parent (an event_id)
                correlation_id  TEXT,       -- trace id across chain

                status     TEXT NOT NULL DEFAULT 'committed', -- received|committed|processed|failed|redacted|superseded
                processed_at REAL,
                error_text TEXT,

                importance_hint INTEGER NOT NULL DEFAULT 0,
                hash TEXT
            );
            """)

            # ---- indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, ts);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_corr_ts ON events(correlation_id, ts);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_event_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_env_ts ON events(environment_id, ts);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_channel_ts ON events(channel, ts);")

            # ---- ensure session exists when using explicit session_id override
            # Without this, inserting into events will fail FK(events.session_id -> sessions.session_id).
            if self._session_id_override:
                sid = str(self._session_id_override)
                cur.execute(
                    """
                    INSERT OR IGNORE INTO sessions (session_id, started_at, channel, environment_id, meta_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        sid,
                        _now_iso(),
                        str(self.default_channel or 'cli'),
                        str(self.default_environment_id or 'local'),
                        "{}",
                    ),
                )
                conn.commit()

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
                event_id  TEXT NOT NULL,
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

            # Auto-migrate older DBs (adds missing event columns/indexes safely)
            self._migrate_events_schema(conn)

    def _migrate_events_schema(self, conn: sqlite3.Connection) -> None:
        """
        Adds missing columns/indexes to events table if DB existed before schema expansion.
        Safe to run repeatedly.
        """
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(events);")
        existing = {row["name"] for row in cur.fetchall()}

        def add_col(name: str, ddl_type: str) -> None:
            if name in existing:
                return
            cur.execute(f"ALTER TABLE events ADD COLUMN {name} {ddl_type};")

        # columns that might be missing in older DBs
        add_col("channel", "TEXT")
        add_col("environment_id", "TEXT")
        add_col("modality", "TEXT")
        add_col("content_format", "TEXT")
        add_col("parent_event_id", "TEXT")
        add_col("correlation_id", "TEXT")
        add_col("status", "TEXT NOT NULL DEFAULT 'committed'")
        add_col("processed_at", "REAL")
        add_col("error_text", "TEXT")

        # indexes (idempotent)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_corr_ts ON events(correlation_id, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_event_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_env_ts ON events(environment_id, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_channel_ts ON events(channel, ts);")

        conn.commit()

    # -------------------------
    # Session helpers
    # -------------------------

    def _get_or_create_current_session(self, conn: sqlite3.Connection) -> str:
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

        session_id = f"sess_{_uuid()}"
        cur.execute(
            """INSERT INTO sessions (session_id, started_at, channel, environment_id, meta_json)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, _now_iso(), self.default_channel, self.default_environment_id, "{}"),
        )
        cur.execute("""
            INSERT INTO state_kv (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
        """, ("current_session_id", _json_dumps(session_id), _now_ts()))
        conn.commit()
        return session_id

    def end_session(self) -> None:
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
        - Stores to canonical events table
        - Returns integer event_seq for compatibility

        New support:
        - channel/environment_id
        - parent_event_id / correlation_id
        - status/processed_at/error_text
        - modality/content_format
        """
        if meta is None:
            meta = {}

        ts = _now_ts()
        ts_iso = _now_iso()

        # normalize actor_type to your DB language
        if role == "npc":
            actor_type = "assistant"
        elif role in ("assistant", "user", "system"):
            actor_type = role
        else:
            actor_type = "unknown"

        payload = dict(meta)
        payload.setdefault("role", role)  # keep compatibility

        # core fields
        event_type = str(payload.get("event_type") or "message")
        channel = str(payload.get("channel") or self.default_channel or "cli")
        environment_id = payload.get("environment_id", self.default_environment_id)
        environment_id = str(environment_id) if environment_id is not None else None

        modality = payload.get("modality")
        modality = str(modality) if modality is not None else ("text" if event_type == "message" else None)

        content_format = payload.get("content_format")
        content_format = str(content_format) if content_format is not None else "plain"

        parent_event_id = payload.get("parent_event_id")
        parent_event_id = str(parent_event_id) if parent_event_id else None

        # ---- correlation policy (simple + useful)
        corr = payload.get("correlation_id")
        corr = str(corr) if corr else None

        status = str(payload.get("status") or "committed")
        processed_at = payload.get("processed_at")
        try:
            processed_at = float(processed_at) if processed_at is not None else None
        except Exception:
            processed_at = None

        error_text = payload.get("error_text")
        error_text = str(error_text) if error_text else None

        with self.connect() as conn:
            # Session scoping:
            # - If a session_id override was provided at DB construction, always use it.
            # - Else fallback to the DB's current/resumable session.
            session_id = self._session_id_override or self._get_or_create_current_session(conn)
            cur = conn.cursor()

            # Ensure the chosen session_id exists in sessions table (FK safety)
            # This is crucial when session_id is provided as an override.
            cur.execute(
                "INSERT OR IGNORE INTO sessions (session_id, started_at, channel, environment_id, meta_json) VALUES (?, ?, ?, ?, ?)",
                (
                    str(session_id),
                    _now_iso(),
                    str(channel or self.default_channel or 'cli'),
                    str(environment_id or self.default_environment_id or 'local'),
                    "{}",
                ),
            )

            # load existing current correlation id if any
            cur.execute("SELECT value_json FROM state_kv WHERE key = ?", ("current_correlation_id",))
            row = cur.fetchone()
            current_corr = None
            if row:
                try:
                    current_corr = json.loads(row["value_json"])
                except Exception:
                    current_corr = None
            if not isinstance(current_corr, str):
                current_corr = None

            # choose correlation id
            if corr:
                correlation_id = corr
            else:
                if actor_type == "user":
                    correlation_id = f"corr_{_uuid()}"
                    # set it as current for the rest of the turn
                    cur.execute("""
                        INSERT INTO state_kv (key, value_json, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                    """, ("current_correlation_id", _json_dumps(correlation_id), _now_ts()))
                else:
                    correlation_id = current_corr

            event_id = f"evt_{_uuid()}"

            # actor_id policy
            actor_id = payload.get("actor_id")
            if actor_id is None and actor_type == "assistant":
                actor_id = self.npc_id
            actor_id = str(actor_id) if actor_id is not None else None

            payload_json = _json_dumps(payload)

            cur.execute(
                """INSERT INTO events
                   (event_id, ts, ts_iso, session_id,
                    channel, environment_id,
                    actor_type, actor_id,
                    event_type, modality, content_format,
                    content, payload_json,
                    parent_event_id, correlation_id,
                    status, processed_at, error_text,
                    importance_hint, hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id, ts, ts_iso, session_id,
                    channel, environment_id,
                    actor_type, actor_id,
                    event_type, modality, content_format,
                    content, payload_json,
                    parent_event_id, correlation_id,
                    status, processed_at, error_text,
                    int(payload.get("importance_hint", 0) or 0),
                    payload.get("hash"),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_recent_events(
        self,
        limit: int = 20,
        *,
        session_id: Optional[str] = None,
        channel: Optional[str] = None,
        environment_id: Optional[str] = None,
        run_across_sessions: Optional[bool] = None,
        run_across_channels: Optional[bool] = None,
        run_across_environments: Optional[bool] = None,
    ) -> List[Event]:
        """
        Fetch recent events, with optional scoping.

        Scoping rules:
        - If run_across_* is True, that dimension is NOT filtered.
        - If run_across_* is False, the dimension is filtered using the provided value,
          or the DB defaults (current session / default channel / default environment).
        """
        # Resolve scope toggles (per-call overrides fall back to constructor defaults)
        across_sessions = self.run_across_sessions if run_across_sessions is None else bool(run_across_sessions)
        across_channels = self.run_across_channels if run_across_channels is None else bool(run_across_channels)
        across_envs = self.run_across_environments if run_across_environments is None else bool(run_across_environments)

        with self.connect() as conn:
            cur = conn.cursor()

            # Determine current session id if needed
            effective_session_id: Optional[str] = None
            if not across_sessions:
                if session_id:
                    effective_session_id = str(session_id)
                else:
                    effective_session_id = self._get_or_create_current_session(conn)

            effective_channel: Optional[str] = None
            if not across_channels:
                effective_channel = str(channel or self.default_channel or "cli")

            effective_env_id: Optional[str] = None
            if not across_envs:
                if environment_id is not None:
                    effective_env_id = str(environment_id)
                else:
                    effective_env_id = str(self.default_environment_id) if self.default_environment_id is not None else None

            where = []
            params: List[Any] = []

            if effective_session_id is not None:
                where.append("session_id = ?")
                params.append(effective_session_id)
            if effective_channel is not None:
                where.append("channel = ?")
                params.append(effective_channel)
            if not across_envs:
                # environment_id can legitimately be NULL; treat None as "only NULL" if default is None
                if effective_env_id is None:
                    where.append("environment_id IS NULL")
                else:
                    where.append("environment_id = ?")
                    params.append(effective_env_id)

            where_sql = ("WHERE " + " AND ".join(where)) if where else ""

            cur.execute(
                f"""SELECT event_seq, ts, actor_type, content, payload_json
                    FROM events
                    {where_sql}
                    ORDER BY event_seq DESC
                    LIMIT ?""",
                tuple(params + [limit]),
            )
            rows = cur.fetchall()

        events: List[Event] = []
        for r in reversed(rows):
            payload = _json_loads(r["payload_json"])

            role = payload.get("role")
            if not role:
                at = str(r["actor_type"])
                role = at if at in ("user", "assistant", "system") else "user"

            # Keep legacy Event signature
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
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM events;")
            cur.execute("DELETE FROM sessions;")
            # also clear correlation/session pointers so you don’t reference ghosts
            cur.execute("DELETE FROM state_kv WHERE key IN ('current_session_id','current_correlation_id');")
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
    # New capabilities
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