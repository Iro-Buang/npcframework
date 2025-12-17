from NPCLoader import load_npc
from NPC_DB_Manager import NPCDatabase

npc = load_npc("npc/kevin.npc")
db = NPCDatabase(npc.paths.db)
db.init_db()

db.add_event("user", "Hello Kevin")
db.add_event("assistant", "Sup. I exist now.")

events = db.get_recent_events(10)
for e in events:
    print(e.role, ":", e.content)

db.set_state("mood", "neutral")
print("mood =", db.get_state("mood"))
