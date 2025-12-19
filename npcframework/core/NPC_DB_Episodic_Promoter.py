from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class PromoteConfig:
    cursor_state_key: str = "episodic_last_event_seq"

    # Chunking rules
    max_events_per_episode: int = 30
    gap_seconds_split: int = 20 * 60  # 20 minutes
    min_events_per_episode: int = 4

    # Optional: only promote message events
    only_event_types: Tuple[str, ...] = ("message",)

    # If True, use sessions.channel/environment_id as authoritative episode fields
    prefer_session_fields: bool = True

    # === FIXES ===
    # Use only USER text to infer intent/entities/tags by default.
    # (Assistant text is often full of system prompt words -> tag pollution)
    analyze_user_only: bool = True

    # Tags: only apply debug/database/memory tags if HARD indicators appear.
    strict_tags: bool = True


class EpisodicPromoter:
    """
    Deterministic event -> episodic memory promoter.
    Uses NPCDatabase.connect() and NPCDatabase.create_memory().

    What it produces (memories.data_json):
      - channel: str
      - environment_id: str
      - intent: str
      - entities: List[str]
      - open_loops: List[str]
      - highlights: List[{who,text}]
      - tags: List[str]
      - stats: counts
    """

    _Q_WORDS = {
        "what", "why", "how", "when", "where", "who", "which",
        "can", "could", "should", "would", "do", "does", "did",
        "is", "are", "was", "were", "am", "will", "may", "might",
    }

    _TASK_WORDS = {
        "build", "make", "implement", "fix", "debug", "write", "create",
        "refactor", "design", "add", "remove", "change", "update",
    }

    # "Hard" debug signals
    _HARD_DEBUG = {
        "traceback", "exception", "stack trace", "stacktrace", "segfault",
        "crash", "failed", "rpc failed", "assert", "panic", "error:",
    }

    # "Hard" database signals
    _HARD_DB = {
        "sqlite", "schema", "table", "query", "index", "pragma",
        "foreign key", "select ", "insert ", "update ", "delete ",
    }

    # "Hard" memory system signals (so "memory" isn't always a tag)
    _HARD_MEMORY = {
        "episodic", "semantic", "memory promotion", "promote memory",
        "retrieval", "recall", "embedding", "vector", "rag",
    }

    _TROLL_WORDS = {"meow", "lol", "lmao", "bruh", "xd", "haha"}

    # entity stopwords (lower)
    _ENTITY_STOP = {
        # conversational junk
        "aight", "alright", "ok", "okay", "hey", "hi", "hello", "yo",
        "nice", "try", "show", "want", "need", "help", "please",

        # pronouns / function words
        "i", "me", "my", "mine", "we", "us", "our", "ours",
        "you", "your", "yours",
        "a", "an", "the", "and", "or", "but",
        "this", "that", "these", "those",
        "now", "then", "just", "well", "so",

        # question starters
        "what", "whats", "why", "how", "when", "where", "who", "which",

        # generic roles
        "npc", "npcs", "assistant", "system", "user",

        # contraction fragments that show up as “Don”
        "don", "dont", "cant", "wont", "im", "ive", "its",
    }

    def __init__(self, db: Any, *, config: Optional[PromoteConfig] = None) -> None:
        self.db = db
        self.cfg = config or PromoteConfig()

        self._re_handle = re.compile(r"@[A-Za-z0-9_]{2,30}")
        self._re_name_1 = re.compile(r"\bmy name is ([A-Z][a-zA-Z0-9_-]{1,30})\b")
        self._re_name_2 = re.compile(r"\bi[' ]?m ([A-Z][a-zA-Z0-9_-]{1,30})\b")
        self._re_name_3 = re.compile(r"\bi am ([A-Z][a-zA-Z0-9_-]{1,30})\b")

        # includes “Kevin”, “Kendrick”
        self._re_cap_token = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
        # includes “NPCFramework”, “ChatGPT”, “LLM”
        self._re_mixedcase = re.compile(r"\b[A-Za-z]+[A-Z][A-Za-z0-9_]*\b")
        self._re_acronym = re.compile(r"\b[A-Z]{2,}\b")

        self._re_has_qword = re.compile(r"\b(" + "|".join(sorted(self._Q_WORDS)) + r")\b", re.IGNORECASE)

        # line start / sentence start heuristic
        self._re_first_word = re.compile(r"^\s*([A-Za-z0-9_@#-]+)")

    # -------------------------
    # Main entry
    # -------------------------

    def promote(self) -> int:
        # print("Promotion started")

        last_seq = self._get_cursor()

        pending: List[int] = self.db.get_state("episodic_pending_event_seqs", []) or []
        if not isinstance(pending, list):
            pending = []

        new_events = self._fetch_events_after(last_seq)
        if not new_events and not pending:
            return 0

        pending.extend([e["event_seq"] for e in new_events])

        # dedupe + keep order
        seen = set()
        pending = [x for x in pending if not (x in seen or seen.add(x))]

        if len(pending) < self.cfg.min_events_per_episode:
            self.db.set_state("episodic_pending_event_seqs", pending)
            # print(f"Not enough events yet ({len(pending)}/{self.cfg.min_events_per_episode}). Pending buffered.")
            return 0

        pending_events = self._fetch_events_by_seq(pending)
        if not pending_events:
            self.db.set_state("episodic_pending_event_seqs", pending)
            return 0

        episodes = self._chunk_events(pending_events)

        created = 0
        used_seqs: List[int] = []

        for chunk in episodes:
            if not chunk or len(chunk) < self.cfg.min_events_per_episode:
                continue

            source_event_seqs = [e["event_seq"] for e in chunk]

            sess_channel, sess_env = self._get_session_fields(chunk[0]["session_id"])
            episode_channel = sess_channel or self._dominant_field(chunk, "channel", default="unknown")
            episode_env = sess_env or self._dominant_field(chunk, "environment_id", default="unknown")

            intent = self._infer_intent(chunk)
            entities = self._extract_entities(chunk)
            open_loops = self._extract_open_loops(chunk)
            highlights = self._extract_highlights(chunk)

            title = self._make_title(chunk, intent, entities)
            summary = self._make_summary(highlights, open_loops, intent)

            tags = self._tag_episode(chunk, intent, episode_channel, episode_env)

            salience = self._compute_salience(chunk, highlights, open_loops, entities)
            confidence = 0.8 if intent in ("task", "debugging", "question") else 0.7

            self.db.create_memory(
                memory_type="episodic",
                title=title,
                summary=summary,
                data={
                    "session_id": chunk[0]["session_id"],
                    "channel": episode_channel,
                    "environment_id": episode_env,

                    "start_event_seq": chunk[0]["event_seq"],
                    "end_event_seq": chunk[-1]["event_seq"],
                    "start_ts": chunk[0]["ts"],
                    "end_ts": chunk[-1]["ts"],

                    "intent": intent,
                    "entities": entities,
                    "open_loops": open_loops,
                    "highlights": highlights,
                    "tags": tags,

                    "stats": {
                        "event_count": len(chunk),
                        "user_count": sum(1 for e in chunk if e["actor_type"] == "user"),
                        "assistant_count": sum(1 for e in chunk if e["actor_type"] in ("assistant", "npc")),
                        "system_count": sum(1 for e in chunk if e["actor_type"] == "system"),
                    },
                },
                salience=salience,
                confidence=confidence,
                scope="npc",
                visibility="private",
                pinned=False,
                source_event_seqs=source_event_seqs,
            )

            created += 1
            # print("Promoted Memory")
            used_seqs.extend(source_event_seqs)

        used = set(used_seqs)
        pending = [x for x in pending if x not in used]

        if new_events:
            self._set_cursor(new_events[-1]["event_seq"])

        self.db.set_state("episodic_pending_event_seqs", pending)
        return created

    # -------------------------
    # Cursor
    # -------------------------

    def _get_cursor(self) -> int:
        raw = self.db.get_state(self.cfg.cursor_state_key, 0)
        try:
            return int(raw)
        except Exception:
            return 0

    def _set_cursor(self, seq: int) -> None:
        self.db.set_state(self.cfg.cursor_state_key, int(seq))

    # -------------------------
    # Fetch
    # -------------------------

    def _fetch_events_by_seq(self, seqs: List[int]) -> List[Dict[str, Any]]:
        if not seqs:
            return []
        q_marks = ",".join(["?"] * len(seqs))
        with self.db.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT event_seq, ts, session_id, actor_type, event_type, content, channel, environment_id
                FROM events
                WHERE event_seq IN ({q_marks})
                ORDER BY event_seq ASC
                """,
                tuple(seqs),
            )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            event_type = str(r["event_type"] or "")
            if self.cfg.only_event_types and event_type not in self.cfg.only_event_types:
                continue
            out.append(
                {
                    "event_seq": int(r["event_seq"]),
                    "ts": float(r["ts"]),
                    "session_id": str(r["session_id"]),
                    "actor_type": str(r["actor_type"]),
                    "event_type": event_type,
                    "content": str(r["content"] or ""),
                    "channel": str(r["channel"] or ""),
                    "environment_id": str(r["environment_id"] or ""),
                }
            )
        return out

    def _fetch_events_after(self, last_seq: int) -> List[Dict[str, Any]]:
        with self.db.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_seq, ts, session_id, actor_type, event_type, content, channel, environment_id
                FROM events
                WHERE event_seq > ?
                ORDER BY event_seq ASC
                """,
                (last_seq,),
            )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            event_type = str(r["event_type"] or "")
            if self.cfg.only_event_types and event_type not in self.cfg.only_event_types:
                continue
            out.append(
                {
                    "event_seq": int(r["event_seq"]),
                    "ts": float(r["ts"]),
                    "session_id": str(r["session_id"]),
                    "actor_type": str(r["actor_type"]),
                    "event_type": event_type,
                    "content": str(r["content"] or ""),
                    "channel": str(r["channel"] or ""),
                    "environment_id": str(r["environment_id"] or ""),
                }
            )
        return out

    def _get_session_fields(self, session_id: str) -> Tuple[str, str]:
        if not self.cfg.prefer_session_fields:
            return ("", "")
        try:
            with self.db.connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT channel, environment_id FROM sessions WHERE session_id = ?", (session_id,))
                row = cur.fetchone()
            if not row:
                return ("", "")
            return (str(row["channel"] or ""), str(row["environment_id"] or ""))
        except Exception:
            return ("", "")

    # -------------------------
    # Chunking
    # -------------------------

    def _chunk_events(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        episodes: List[List[Dict[str, Any]]] = []
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for e in events:
            buckets.setdefault(e["session_id"], []).append(e)
        for _, evs in buckets.items():
            episodes.extend(self._chunk_within_session(evs))
        return episodes

    def _chunk_within_session(self, evs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        prev_ts: Optional[float] = None

        for e in evs:
            if not current:
                current = [e]
                prev_ts = e["ts"]
                continue

            gap = float(e["ts"]) - float(prev_ts) if prev_ts is not None else 0.0
            if gap >= self.cfg.gap_seconds_split:
                chunks.append(current)
                current = [e]
                prev_ts = e["ts"]
                continue

            current.append(e)
            prev_ts = e["ts"]

            if len(current) >= self.cfg.max_events_per_episode:
                chunks.append(current)
                current = []
                prev_ts = None

        if current:
            chunks.append(current)

        return chunks

    # -------------------------
    # Analysis helpers
    # -------------------------

    def _analysis_text(self, chunk: List[Dict[str, Any]]) -> str:
        if self.cfg.analyze_user_only:
            return " ".join((e.get("content") or "") for e in chunk if e.get("actor_type") == "user").lower()
        return " ".join((e.get("content") or "") for e in chunk).lower()

    # -------------------------
    # Intelligence (deterministic)
    # -------------------------

    def _infer_intent(self, chunk: List[Dict[str, Any]]) -> str:
        text = self._analysis_text(chunk)

        if "?" in text or self._re_has_qword.search(text):
            return "question"

        if any(w in text for w in self._HARD_DEBUG) or any(w in text for w in self._HARD_DB):
            return "debugging"

        if any(w in text for w in self._TASK_WORDS):
            return "task"

        if any(w in text for w in self._TROLL_WORDS):
            return "banter"

        if any(g in text for g in ["hello", "hi", "hey", "ohayo", "good morning", "good evening"]):
            return "greeting"

        return "chitchat"

    def _extract_entities(self, chunk: List[Dict[str, Any]]) -> List[str]:
        # key fix: extract from USER text (unless disabled)
        if self.cfg.analyze_user_only:
            lines = [(e.get("content") or "").strip() for e in chunk if e.get("actor_type") == "user" and (e.get("content") or "").strip()]
        else:
            lines = [(e.get("content") or "").strip() for e in chunk if (e.get("content") or "").strip()]

        if not lines:
            return []

        raw: List[str] = []

        # explicit name patterns
        for line in lines:
            for pat in (self._re_name_1, self._re_name_2, self._re_name_3):
                m = pat.search(line)
                if m:
                    raw.append(m.group(1))

        # handles
        for line in lines:
            raw.extend(self._re_handle.findall(line))

        # mixedcase brands / acronyms
        for line in lines:
            raw.extend(self._re_mixedcase.findall(line))
            raw.extend(self._re_acronym.findall(line))

        # proper nouns, but filter sentence-start hype words
        for line in lines:
            first = ""
            m = self._re_first_word.search(line)
            if m:
                first = m.group(1)

            for t in self._re_cap_token.findall(line):
                tl = t.lower()
                if tl in self._ENTITY_STOP:
                    continue
                if first and t == first and tl in self._ENTITY_STOP:
                    continue
                if first and t == first and tl in self._Q_WORDS:
                    continue
                raw.append(t)

        out: List[str] = []
        seen = set()
        for x in raw:
            x2 = x.strip()
            if not x2:
                continue
            if x2.lower() in self._ENTITY_STOP:
                continue
            if x2 not in seen:
                seen.add(x2)
                out.append(x2)

        return out[:12]

    def _extract_open_loops(self, chunk: List[Dict[str, Any]]) -> List[str]:
        user_msgs = [
            (e.get("content") or "").strip()
            for e in chunk
            if e.get("actor_type") == "user" and (e.get("content") or "").strip()
        ]
        asst_text = " ".join(
            (e.get("content") or "")
            for e in chunk
            if e.get("actor_type") in ("assistant", "npc") and (e.get("content") or "")
        ).lower()

        candidates: List[str] = []
        for u in user_msgs:
            ul = u.lower()
            if "?" in u:
                candidates.append(self._one_line(u))
            elif any(w in ul for w in self._TASK_WORDS) and len(u) > 12:
                candidates.append(self._one_line(u))

        resolved_markers = ["here's", "do this", "replace", "change", "fix", "solution", "steps", "use this", "patch"]
        resolved = any(m in asst_text for m in resolved_markers)

        if resolved:
            return candidates[:1]
        return candidates[:3]

    def _score_line(self, who: str, text: str) -> int:
        t = text.strip()
        tl = t.lower()
        if not t:
            return 0

        s = 0
        if "?" in t:
            s += 4
        if any(w in tl for w in self._HARD_DEBUG):
            s += 4
        if any(w in tl for w in self._TASK_WORDS):
            s += 3
        if any(w in tl for w in ["remember", "important", "note", "save", "later", "tomorrow", "next time"]):
            s += 3
        if len(t) >= 80:
            s += 2
        if who == "user":
            s += 1

        if tl in {"ok", "okay", "k", "lol", "lmao", "meow", "meowww"}:
            s -= 3
        if len(t) <= 6 and any(w in tl for w in self._TROLL_WORDS):
            s -= 2

        return s

    def _extract_highlights(self, chunk: List[Dict[str, Any]], max_items: int = 8) -> List[Dict[str, str]]:
        items: List[Tuple[int, str, str]] = []
        for e in chunk:
            txt = (e.get("content") or "").strip()
            if not txt:
                continue
            who = "assistant" if e.get("actor_type") in ("assistant", "npc") else str(e.get("actor_type") or "unknown")
            items.append((self._score_line(who, txt), who, self._one_line(txt)))

        if not items:
            return []

        picks: List[Tuple[str, str]] = []

        user_lines = [(who, txt) for _, who, txt in items if who == "user"]
        if user_lines:
            picks.append(user_lines[0])
            if user_lines[-1] != user_lines[0]:
                picks.append(user_lines[-1])

        best_asst: Optional[Tuple[str, str]] = None
        best_score = -10**9
        for score, who, txt in items:
            if who == "assistant" and score > best_score:
                best_score = score
                best_asst = (who, txt)
        if best_asst:
            picks.append(best_asst)

        items.sort(key=lambda x: x[0], reverse=True)
        for score, who, txt in items:
            if score <= 0:
                continue
            if (who, txt) not in picks:
                picks.append((who, txt))
            if len(picks) >= max_items:
                break

        out: List[Dict[str, str]] = []
        seen = set()
        for who, txt in picks:
            key = (who, txt)
            if key in seen:
                continue
            seen.add(key)
            out.append({"who": who, "text": txt})
        return out[:max_items]

    def _compute_salience(self, chunk: List[Dict[str, Any]], highlights: List[Dict[str, str]], open_loops: List[str], entities: List[str]) -> int:
        text = self._analysis_text(chunk)
        s = 30

        s += min(20, 2 * len(highlights))
        s += min(20, 6 * len(open_loops))
        s += min(15, 3 * len(entities))

        if any(w in text for w in self._HARD_DEBUG):
            s += 20
        if any(w in text for w in self._TASK_WORDS):
            s += 15
        if "?" in text:
            s += 10

        return max(0, min(100, s))

    # -------------------------
    # Title / Summary / Tags
    # -------------------------

    def _make_title(self, chunk: List[Dict[str, Any]], intent: str, entities: List[str]) -> str:
        for e in chunk:
            if e.get("actor_type") == "user":
                txt = (e.get("content") or "").strip()
                if not txt or txt.startswith("/"):
                    continue
                if "?" in txt or any(w in txt.lower() for w in self._TASK_WORDS):
                    return (txt[:60] + "…") if len(txt) > 60 else txt

        if entities:
            return f"{intent}: {entities[0]}"
        return f"{intent}: {chunk[0]['event_seq']}-{chunk[-1]['event_seq']}"

    def _make_summary(self, highlights: List[Dict[str, str]], open_loops: List[str], intent: str) -> str:
        parts: List[str] = []
        if highlights:
            h = " | ".join([f"{x['who']}: {x['text']}" for x in highlights[:4]])
            parts.append(f"Highlights: {h}")
        if open_loops:
            parts.append("Open loops: " + " | ".join(open_loops))
        parts.append(f"Intent: {intent}.")
        return " ".join(parts)

    def _tag_episode(self, chunk: List[Dict[str, Any]], intent: str, channel: str, env: str) -> List[str]:
        text = self._analysis_text(chunk)  # USER-only by default

        tags: List[str] = []
        if intent != "chitchat":
            tags.append(intent)

        hard_debug = any(w in text for w in self._HARD_DEBUG)
        hard_db = any(w in text for w in self._HARD_DB)
        hard_mem = any(w in text for w in self._HARD_MEMORY)

        if self.cfg.strict_tags:
            if intent == "debugging" or hard_debug:
                tags.append("debugging")
            if hard_db:
                tags.append("database")
            if hard_mem:
                tags.append("memory")
        else:
            if hard_debug:
                tags.append("debugging")
            if hard_db:
                tags.append("database")
            if "memory" in text:
                tags.append("memory")

        if channel and channel != "unknown":
            tags.append(f"channel:{channel}")
        if env and env != "unknown":
            tags.append(f"env:{env}")

        return self._dedupe(tags) or ["misc"]

    # -------------------------
    # Utility
    # -------------------------

    def _one_line(self, s: str) -> str:
        s = s.replace("\n", " ").replace("\r", " ").strip()
        return (s[:120] + "…") if len(s) > 120 else s

    def _dominant_field(self, chunk: List[Dict[str, Any]], key: str, default: str = "") -> str:
        counts: Dict[str, int] = {}
        for e in chunk:
            v = (e.get(key) or "").strip()
            if not v:
                continue
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            return default
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _dedupe(self, xs: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in xs:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out
