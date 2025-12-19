from __future__ import annotations

import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from NPCLoader import load_npc
from NPC_DB_Manager import NPCDatabase
from NPCPrompt_Compiler import compile_messages, CompileOptions
from npcframework.inference.llamacpp import LlamaCppEngine, LlamaCppConfig


# -----------------------------
# Shared helpers
# -----------------------------

HARD_STOP_TOKENS = (
    "/stop",
    "/end",
    "[stop]",
)

BANTER_BEATS: List[str] = [
    "Playful bickering (no real stakes, just sport).",
    "Petty rivalry (one-up each other, lightly).",
    "Domestic logistics (tiny decisions treated like life-or-death).",
    "Philosophy-but-meme (existential, but annoying about it).",
    "Taste war (food, music, habits, preferences).",
    "Tech bruh debate (framework vs. practicality).",
    "Condo vs countryside (lifestyle and identity).",
    "‘What if’ scenario (ridiculous but argued seriously).",
]

TOPIC_POOL: List[str] = [
    "Which is worse: slow Wi-Fi or a slow person?",
    "Is minimalism disciplined or just being broke with branding?",
    "Condo life vs countryside life—what actually makes you happier?",
    "Should NPCs have ‘feelings’ or just better error handling?",
    "Is productivity a scam or a skill issue?",
    "Do you trust instincts or spreadsheets more?",
    "What’s the most overrated ‘adulting’ milestone?",
    "Is it better to be feared, loved, or simply left alone?",
    "Are humans just complicated routines with snacks?",
    "Which is more annoying: optimism or cynicism?",
]

def _ensure_default_state(db: NPCDatabase) -> None:
    if db.get_state("mode") is None:
        db.set_state("mode", "idle")
    if db.get_state("mood") is None:
        db.set_state("mood", "neutral")
    if db.get_state("energy") is None:
        db.set_state("energy", 0.8)

def _state_snapshot(db: NPCDatabase) -> Dict[str, Any]:
    return {
        "mode": db.get_state("mode", "idle"),
        "mood": db.get_state("mood", "neutral"),
        "energy": db.get_state("energy", 0.8),
    }

def _sanitize_user_input(text: str, npc_names: List[str]) -> str:
    t = text.strip()
    tl = t.lower()
    for name in npc_names:
        prefix = f"{name.lower()}>"
        if tl.startswith(prefix):
            return t.split(">", 1)[1].strip()
    return t

def _compile_for(npc, db: NPCDatabase, history_limit: int, *, relay: Optional[str] = None) -> List[Dict[str, str]]:
    recent = db.get_recent_events(limit=history_limit)
    state = _state_snapshot(db)

    messages = compile_messages(
        identity=npc.identity,
        persona=npc.persona,
        policy=npc.policy,
        recent_events=recent,
        state_snapshot=state,
        options=CompileOptions(history_limit=history_limit, include_state=True),
    )

    if relay:
        messages.append({"role": "system", "content": relay})

    return messages

def _stream_reply(engine: LlamaCppEngine, messages: List[Dict[str, str]], *, show_wait_after: float = 0.9) -> str:
    chunks: List[str] = []
    t0 = time.time()
    showed_wait = False

    for piece in engine.chat_stream(messages):
        if not showed_wait and (time.time() - t0) > show_wait_after and not chunks:
            print("...", end="", flush=True)
            showed_wait = True
        print(piece, end="", flush=True)
        chunks.append(piece)

    return "".join(chunks).strip()

def _contains_hard_stop(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(tok in t for tok in HARD_STOP_TOKENS)

def _pick_next_topic(prev_topic: str) -> str:
    # Prefer variety: avoid repeating the same topic
    candidates = [t for t in TOPIC_POOL if t != prev_topic]
    return random.choice(candidates) if candidates else prev_topic


# -----------------------------
# Duo runtime
# -----------------------------

@dataclass
class NPCSession:
    npc: Any
    db: NPCDatabase
    name: str
    npc_id: str

def make_engine() -> LlamaCppEngine:
    return LlamaCppEngine(
        LlamaCppConfig(
            model_path="inference/models/gemma-3-4b-it-q4_0.gguf",
            n_ctx=4096,
            max_tokens=256,
            temperature=0.85,   # <- slightly higher for banter/creativity
            top_p=0.92,
            n_gpu_layers=0,
        )
    )

def _banter_relay(
    *,
    topic: str,
    beat: str,
    speaker: NPCSession,
    listener: NPCSession,
    last_from_listener: str,
    turn_index: int,
    max_turns: int,
    topic_lock: bool,
) -> str:
    clipped = (last_from_listener or "").strip()
    if len(clipped) > 650:
        clipped = clipped[:650] + "…"

    # topic_lock=False means they are allowed/encouraged to pivot to a new topic.
    pivot_rule = (
        "- If the conversation stalls or you’ve exhausted the topic, pivot to a NEW topic yourself (no permission needed).\n"
        "- When you pivot, start the line with: [NEW_TOPIC] <your new topic>\n"
        if not topic_lock else
        "- Stay on the given topic for now; do not pivot yet.\n"
    )

    return (
        "DUO_BANTER_MODE v1\n"
        f"current_topic: {topic}\n"
        f"banter_beat: {beat}\n"
        f"turn: {turn_index + 1}/{max_turns}\n"
        f"speaker_name: {speaker.name}\n"
        f"listener_name: {listener.name}\n"
        "rules:\n"
        "- Stay in character.\n"
        "- Keep replies 1–6 sentences.\n"
        "- Your default stance is playful disagreement.\n"
        "- Tease lightly; no cruelty.\n"
        "- Do not ‘wrap up’ the conversation unless explicitly instructed by the USER.\n"
        "- Never output [END]. (Not a thing in this mode.)\n"
        f"{pivot_rule}"
        "- If you want the USER to intervene, ask them a direct question.\n"
        "context_from_other:\n"
        f"{listener.name}: {clipped if clipped else '(no prior message)'}\n"
    )

def duo_turn(
    engine: LlamaCppEngine,
    speaker: NPCSession,
    listener: NPCSession,
    *,
    topic: str,
    beat: str,
    last_from_listener: str,
    history_limit: int,
    turn_index: int,
    max_turns: int,
    topic_lock: bool,
) -> str:
    relay = _banter_relay(
        topic=topic,
        beat=beat,
        speaker=speaker,
        listener=listener,
        last_from_listener=last_from_listener,
        turn_index=turn_index,
        max_turns=max_turns,
        topic_lock=topic_lock,
    )

    messages = _compile_for(speaker.npc, speaker.db, history_limit, relay=relay)

    print(f"{speaker.name}> ", end="", flush=True)
    reply = _stream_reply(engine, messages)
    print("\n")

    if reply:
        speaker.db.add_event("assistant", reply)
    else:
        speaker.db.add_event("assistant", "(no output)")

    return reply

def _extract_new_topic(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:4]:  # look near the top
        if ln.lower().startswith("[new_topic]"):
            return ln.split("]", 1)[1].strip()
    return None

def duo_chat_loop(
    left_dir: str,
    right_dir: str,
    *,
    history_limit: int = 20,
    turns_per_duo: int = 10,
    beat_every: int = 8,
) -> None:
    left_npc = load_npc(left_dir)
    right_npc = load_npc(right_dir)

    left_db = NPCDatabase(left_npc.paths.db)
    right_db = NPCDatabase(right_npc.paths.db)
    left_db.init_db()
    right_db.init_db()

    _ensure_default_state(left_db)
    _ensure_default_state(right_db)

    left = NPCSession(
        npc=left_npc,
        db=left_db,
        name=left_npc.manifest.get("display_name", left_npc.manifest.get("id", "LeftNPC")),
        npc_id=left_npc.manifest.get("id", "left"),
    )
    right = NPCSession(
        npc=right_npc,
        db=right_db,
        name=right_npc.manifest.get("display_name", right_npc.manifest.get("id", "RightNPC")),
        npc_id=right_npc.manifest.get("id", "right"),
    )

    engine = make_engine()

    starter_left = True
    npc_names = [left.name, right.name]

    print(f"{left.name} online. NPC={left.npc_id}  DB={left_npc.paths.db}")
    print(f"{right.name} online. NPC={right.npc_id}  DB={right_npc.paths.db}")
    print("\nCommands:")
    print("  /duo <topic>     Start banter on a topic")
    print("  /random          Start banter on a random topic")
    print("  /turns <N>       Set turns (default 10)  NOTE: turns are pairs => total NPC replies = N*2")
    print("  /swap            Swap who starts")
    print("  /beat <N>        Change beat cadence (default 8 turns)")
    print("  /stop            Stop the current duo run (or /exit to quit)")
    print("  /exit            Quit")
    print("\nTip: You can type 'Anna> blah' or 'Kevin> blah' and I’ll strip the prefix.\n")

    while True:
        raw = input("You> ")
        if raw is None:
            continue

        text = _sanitize_user_input(raw, npc_names)
        if not text:
            continue

        if text.startswith("/exit"):
            break

        if text.startswith("/swap"):
            starter_left = not starter_left
            print(f"(starter is now {'left' if starter_left else 'right'})\n")
            continue

        if text.startswith("/turns"):
            parts = text.split()
            if len(parts) == 2 and parts[1].isdigit():
                turns_per_duo = max(1, int(parts[1]))
                print(f"(turns_per_duo set to {turns_per_duo})\n")
            else:
                print("(usage: /turns 10)\n")
            continue

        if text.startswith("/beat"):
            parts = text.split()
            if len(parts) == 2 and parts[1].isdigit():
                beat_every = max(1, int(parts[1]))
                print(f"(beat cadence set to every {beat_every} turns)\n")
            else:
                print("(usage: /beat 8)\n")
            continue

        if text.startswith("/random"):
            topic = random.choice(TOPIC_POOL)
            text = f"/duo {topic}"

        if text.startswith("/duo"):
            topic = text[len("/duo"):].strip()
            if not topic:
                print("(usage: /duo <topic>)\n")
                continue

            a = left if starter_left else right
            b = right if starter_left else left

            # log once
            a.db.add_event("user", f"[DUO_BANTER_TOPIC] {topic}")
            b.db.add_event("user", f"[DUO_BANTER_TOPIC] {topic}")

            last_a = ""
            last_b = ""

            max_turns = turns_per_duo * 2
            current_topic = topic

            for t in range(max_turns):
                beat = BANTER_BEATS[(t // beat_every) % len(BANTER_BEATS)]

                # Lock topic for first few turns so they don't instantly pivot
                topic_lock = t < 4

                if t % 2 == 0:
                    last_a = duo_turn(
                        engine, a, b,
                        topic=current_topic,
                        beat=beat,
                        last_from_listener=last_b,
                        history_limit=history_limit,
                        turn_index=t,
                        max_turns=max_turns,
                        topic_lock=topic_lock,
                    )
                    if _contains_hard_stop(last_a):
                        break

                    new_topic = _extract_new_topic(last_a)
                    if new_topic and not topic_lock:
                        current_topic = new_topic
                        # log topic shift (optional)
                        a.db.add_event("user", f"[TOPIC_SHIFT] {current_topic}")
                        b.db.add_event("user", f"[TOPIC_SHIFT] {current_topic}")

                else:
                    last_b = duo_turn(
                        engine, b, a,
                        topic=current_topic,
                        beat=beat,
                        last_from_listener=last_a,
                        history_limit=history_limit,
                        turn_index=t,
                        max_turns=max_turns,
                        topic_lock=topic_lock,
                    )
                    if _contains_hard_stop(last_b):
                        break

                    new_topic = _extract_new_topic(last_b)
                    if new_topic and not topic_lock:
                        current_topic = new_topic
                        a.db.add_event("user", f"[TOPIC_SHIFT] {current_topic}")
                        b.db.add_event("user", f"[TOPIC_SHIFT] {current_topic}")

            continue

        # Single mode fallback: talk to starter
        target = left if starter_left else right
        target.db.add_event("user", text)
        msgs = _compile_for(
            target.npc,
            target.db,
            history_limit,
            relay=(
                "SINGLE_MODE:\n"
                "- Respond to the user.\n"
                "- Keep it concise.\n"
            ),
        )
        print(f"{target.name}> ", end="", flush=True)
        try:
            reply = _stream_reply(engine, msgs)
            print("\n")
            target.db.add_event("assistant", reply or "(no output)")
        except KeyboardInterrupt:
            print(f"\n{target.name}> (interrupted)\n")
        except Exception as e:
            print(f"\n{target.name}> (inference error: {e})\n")


if __name__ == "__main__":
    left_dir = sys.argv[1] if len(sys.argv) > 1 else "npc/kevin.npc"
    right_dir = sys.argv[2] if len(sys.argv) > 2 else "npc/anna.npc"
    duo_chat_loop(left_dir, right_dir, history_limit=20, turns_per_duo=10)
