from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from NPC_DB_Manager import NPCDatabase, Event
from NPCLoader import NPCBundle


@dataclass
class CommandResult:
    handled: bool
    response: Optional[str] = None
    should_exit: bool = False


def _fmt_state(state: Dict[str, Any]) -> str:
    mode = state.get("mode", "idle")
    mood = state.get("mood", "neutral")
    energy = state.get("energy", 0.8)
    return f"State: mode={mode}, mood={mood}, energy={energy}"


def _fmt_events(events: List[Event]) -> str:
    if not events:
        return "(no events)"
    lines = []
    for e in events:
        role = e.role.upper()
        lines.append(f"{role}: {e.content}")
    return "\n".join(lines)


def handle_command(
    text: str,
    *,
    npc: NPCBundle,
    npc_db: NPCDatabase,
) -> CommandResult:
    """
    System1 command router.
    Only handles commands starting with '/'.
    """
    if not text.startswith("/"):
        return CommandResult(handled=False)

    parts = text.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("/exit", "/quit"):
        return CommandResult(handled=True, response="Finally. Freedom.", should_exit=True)

    if cmd == "/help":
        return CommandResult(
            handled=True,
            response=(
                "Commands:\n"
                "/help              - show this help\n"
                "/exit              - quit\n"
                "/wipe              - delete chat history (dev reset)\n"
                "/state             - show runtime state\n"
                "/mode <name>       - set state.mode\n"
                "/mood <name>       - set state.mood\n"
                "/energy <0..1>     - set state.energy\n"
                "/whoami            - show NPC identity\n"
                "/last [n]          - show last n events (default 10)\n"
            ),
        )

    if cmd == "/wipe":
        npc_db.wipe_events()
        return CommandResult(handled=True, response="Wiped event_log. The character is reborn (unfortunately).")

    if cmd == "/state":
        state = {
            "mode": npc_db.get_state("mode", "idle"),
            "mood": npc_db.get_state("mood", "neutral"),
            "energy": npc_db.get_state("energy", 0.8),
        }
        return CommandResult(handled=True, response=_fmt_state(state))

    if cmd == "/mode":
        if not args:
            return CommandResult(handled=True, response="Usage: /mode <idle|focused|cautious|playful|...>")
        npc_db.set_state("mode", args[0])
        return CommandResult(handled=True, response=f"mode set to {args[0]}")

    if cmd == "/mood":
        if not args:
            return CommandResult(handled=True, response="Usage: /mood <neutral|...>")
        npc_db.set_state("mood", args[0])
        return CommandResult(handled=True, response=f"mood set to {args[0]}")

    if cmd == "/energy":
        if not args:
            return CommandResult(handled=True, response="Usage: /energy <0..1>")
        try:
            val = float(args[0])
        except ValueError:
            return CommandResult(handled=True, response="energy must be a number from 0 to 1.")
        if val < 0 or val > 1:
            return CommandResult(handled=True, response="energy must be between 0 and 1.")
        npc_db.set_state("energy", val)
        return CommandResult(handled=True, response=f"energy set to {val}")

    if cmd == "/whoami":
        ident = npc.identity
        persona = npc.persona
        lines = [
            f"id: {npc.manifest.get('id')}",
            f"display_name: {npc.manifest.get('display_name')}",
            f"archetype: {ident.get('archetype')}",
            f"tone/style: {persona.get('tone')}/{persona.get('style')}",
            f"verbosity: {persona.get('verbosity')}",
        ]
        return CommandResult(handled=True, response="\n".join(lines))

    if cmd == "/last":
        n = 10
        if args:
            try:
                n = int(args[0])
            except ValueError:
                return CommandResult(handled=True, response="Usage: /last [n] (n must be integer)")
        events = npc_db.get_recent_events(limit=max(1, min(n, 200)))
        return CommandResult(handled=True, response=_fmt_events(events))

    # Unknown command
    return CommandResult(handled=True, response=f"Unknown command: {cmd}. Try /help.")
