from npcframework.core.NPC_Loader import load_npc

npc = load_npc("../../npc/kevin.npc")
print(npc.manifest["id"], npc.persona["tone"], npc.paths.db)
print(npc.policy)

print(npc.identity)