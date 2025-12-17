from NPCLoader import load_npc
from db import NPCDatabase
from NPCPrompt_Compiler import compile_messages, CompileOptions


def main():
    npc = load_npc("npc/kevin.npc")

    db = NPCDatabase(npc.paths.db)
    db.init_db()

    # OPTIONAL: seed only if DB is empty
    recent = db.get_recent_events(limit=5)
    if not recent:
        db.add_event("user", "Hello Kevin")
        db.add_event("assistant", "Sup. I exist now.")

    # state defaults
    if db.get_state("mode") is None:
        db.set_state("mode", "idle")
    if db.get_state("mood") is None:
        db.set_state("mood", "neutral")
    if db.get_state("energy") is None:
        db.set_state("energy", 0.8)

    recent = db.get_recent_events(limit=20)

    state_snapshot = {
        "mode": db.get_state("mode"),
        "mood": db.get_state("mood"),
        "energy": db.get_state("energy"),
    }

    messages = compile_messages(
        identity=npc.identity,
        persona=npc.persona,
        policy=npc.policy,
        recent_events=recent,
        state_snapshot=state_snapshot,
        options=CompileOptions(history_limit=20, include_state=True),
    )

    print("\n===== COMPILED MESSAGES =====\n")
    for i, m in enumerate(messages, 1):
        print(f"[{i}] {m['role'].upper()}")
        print(m["content"])
        print("-" * 60)


if __name__ == "__main__":
    main()
