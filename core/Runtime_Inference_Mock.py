from __future__ import annotations

from core.NPC_Loader import load_npc
from NPC_DB_Manager import NPCDatabase
from Runtime_Prompt_Compiler import compile_messages, CompileOptions
from inference.mock import MockEngine
from Runtime_Commands import handle_command


def run_cli(npc_dir: str, *, history_limit: int = 20) -> None:
    npc = load_npc(npc_dir)

    db = NPCDatabase(npc.paths.db)
    db.init_db()

    engine = MockEngine()

    print(f"Kevin online. NPC={npc.manifest.get('id')}  DB={npc.paths.db}")
    print("Type /help for commands.\n")

    while True:
        user_input = input("You> ").strip()
        if not user_input:
            continue

        # 0) System1: command routing
        cmdres = handle_command(user_input, npc=npc, npc_db=db)
        if cmdres.handled:
            if cmdres.response:
                print(f"Kevin> {cmdres.response}\n")
            if cmdres.should_exit:
                break
            continue

        # 1) log user input FIRST (DB is truth)
        db.add_event("user", user_input)

        # 2) state snapshot (tiny)
        state_snapshot = {
            "mode": db.get_state("mode", "idle"),
            "mood": db.get_state("mood", "neutral"),
            "energy": db.get_state("energy", 0.8),
        }

        # 3) fetch recent events (includes the user message we just logged)
        recent = db.get_recent_events(limit=history_limit)

        # 4) compile messages
        messages = compile_messages(
            identity=npc.identity,
            persona=npc.persona,
            policy=npc.policy,
            recent_events=recent,
            state_snapshot=state_snapshot,
            options=CompileOptions(history_limit=history_limit, include_state=True),
        )

        # 5) infer
        reply = engine.chat(messages)

        # 6) log assistant
        db.add_event("npc", reply)

        print(f"Kevin> {reply}\n")
