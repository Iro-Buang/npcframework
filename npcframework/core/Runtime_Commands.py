from __future__ import annotations

"""
NPCFramework - Runtime Commands (System1)

PURPOSE
- Deterministic command router for CLI/server environments.
- Handles only inputs starting with '/' (no LLM involvement).

PRIMARY ENTRYPOINT
- handle_command(text: str, npc: NPCBundle, npc_db: NPCDatabase) -> CommandResult

I/O CONTRACT
Input:
- text: raw user input
- npc: loaded NPC bundle (manifest/identity/persona/etc.)
- npc_db: NPCDatabase for state and event_log operations

Output:
- CommandResult(handled: bool, response: Optional[str], should_exit: bool)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .NPC_DB_Manager import NPCDatabase, Event
from .NPC_Loader import NPCBundle


# =============================================================================
# CONFIG / CONSTANTS
# =============================================================================

CMD_PREFIX = "/"

DEFAULT_STATE_MODE = "idle"
DEFAULT_STATE_MOOD = "neutral"
DEFAULT_STATE_ENERGY = 0.8

DEFAULT_LAST_N = 10
MAX_LAST_N = 200
MIN_LAST_N = 1

EXIT_COMMANDS = {"/exit", "/quit"}

HELP_LINES = [
    ("/help", "show this help"),
    ("/exit", "quit"),
    ("/wipe", "delete chat history (dev reset)"),
    ("/state", "show runtime state"),
    ("/mode <name>", "set state.mode"),
    ("/mood <name>", "set state.mood"),
    ("/energy <0..1>", "set state.energy"),
    ("/whoami", "show NPC identity"),
    ("/last [n]", "show last n events (default 10)"),
]

# Flavor text. Because you’re Kevin. Not Clippy.
MSG_EXIT = "Finally. Freedom."
MSG_WIPE = "Wiped event_log. The character is reborn (unfortunately)."


# =============================================================================
# TYPES
# =============================================================================

@dataclass
class CommandResult:
    handled: bool
    response: Optional[str] = None
    should_exit: bool = False


# =============================================================================
# FORMATTERS
# =============================================================================

def _fmt_state(state: Dict[str, Any]) -> str:
    mode = state.get("mode", DEFAULT_STATE_MODE)
    mood = state.get("mood", DEFAULT_STATE_MOOD)
    energy = state.get("energy", DEFAULT_STATE_ENERGY)
    return f"State: mode={mode}, mood={mood}, energy={energy}"


def _fmt_events(events: List[Event]) -> str:
    if not events:
        return "(no events)"
    return "\n".join(f"{e.role.upper()}: {e.content}" for e in events)


def _fmt_help() -> str:
    # Keep help aligned-ish without being precious about formatting.
    pad = max(len(cmd) for cmd, _ in HELP_LINES)
    lines = ["Commands:"]
    for cmd, desc in HELP_LINES:
        lines.append(f"{cmd.ljust(pad)}  - {desc}")
    return "\n".join(lines)


# =============================================================================
# STATE HELPERS
# =============================================================================

def _get_state_snapshot(db: NPCDatabase) -> Dict[str, Any]:
    return {
        "mode": db.get_state("mode", DEFAULT_STATE_MODE),
        "mood": db.get_state("mood", DEFAULT_STATE_MOOD),
        "energy": db.get_state("energy", DEFAULT_STATE_ENERGY),
    }


def _set_state_mode(db: NPCDatabase, value: str) -> str:
    db.set_state("mode", value)
    return f"mode set to {value}"


def _set_state_mood(db: NPCDatabase, value: str) -> str:
    db.set_state("mood", value)
    return f"mood set to {value}"


def _set_state_energy(db: NPCDatabase, raw: str) -> str:
    try:
        val = float(raw)
    except ValueError:
        return "energy must be a number from 0 to 1."
    if val < 0 or val > 1:
        return "energy must be between 0 and 1."
    db.set_state("energy", val)
    return f"energy set to {val}"


# =============================================================================
# COMMAND HANDLERS
# =============================================================================

def _cmd_exit(_: List[str], __: NPCBundle, ___: NPCDatabase) -> CommandResult:
    return CommandResult(handled=True, response=MSG_EXIT, should_exit=True)


def _cmd_help(_: List[str], __: NPCBundle, ___: NPCDatabase) -> CommandResult:
    return CommandResult(handled=True, response=_fmt_help())


def _cmd_wipe(_: List[str], __: NPCBundle, db: NPCDatabase) -> CommandResult:
    db.wipe_events()
    return CommandResult(handled=True, response=MSG_WIPE)


def _cmd_state(_: List[str], __: NPCBundle, db: NPCDatabase) -> CommandResult:
    return CommandResult(handled=True, response=_fmt_state(_get_state_snapshot(db)))


def _cmd_mode(args: List[str], __: NPCBundle, db: NPCDatabase) -> CommandResult:
    if not args:
        return CommandResult(handled=True, response="Usage: /mode <idle|focused|cautious|playful|...>")
    return CommandResult(handled=True, response=_set_state_mode(db, args[0]))


def _cmd_mood(args: List[str], __: NPCBundle, db: NPCDatabase) -> CommandResult:
    if not args:
        return CommandResult(handled=True, response="Usage: /mood <neutral|...>")
    return CommandResult(handled=True, response=_set_state_mood(db, args[0]))


def _cmd_energy(args: List[str], __: NPCBundle, db: NPCDatabase) -> CommandResult:
    if not args:
        return CommandResult(handled=True, response="Usage: /energy <0..1>")
    return CommandResult(handled=True, response=_set_state_energy(db, args[0]))


def _cmd_whoami(_: List[str], npc: NPCBundle, __: NPCDatabase) -> CommandResult:
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


def _cmd_last(args: List[str], __: NPCBundle, db: NPCDatabase) -> CommandResult:
    n = DEFAULT_LAST_N
    if args:
        try:
            n = int(args[0])
        except ValueError:
            return CommandResult(handled=True, response="Usage: /last [n] (n must be integer)")

    n = max(MIN_LAST_N, min(n, MAX_LAST_N))
    events = db.get_recent_events(limit=n)
    return CommandResult(handled=True, response=_fmt_events(events))


# Map command string -> handler
COMMAND_TABLE = {
    "/help": _cmd_help,
    "/wipe": _cmd_wipe,
    "/state": _cmd_state,
    "/mode": _cmd_mode,
    "/mood": _cmd_mood,
    "/energy": _cmd_energy,
    "/whoami": _cmd_whoami,
    "/last": _cmd_last,
    "/exit": _cmd_exit,
    "/quit": _cmd_exit,
}


# =============================================================================
# PUBLIC ENTRYPOINT
# =============================================================================

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
    if not text or not text.startswith(CMD_PREFIX):
        return CommandResult(handled=False)

    parts = text.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    handler = COMMAND_TABLE.get(cmd)
    if handler:
        return handler(args, npc, npc_db)

    return CommandResult(handled=True, response=f"Unknown command: {cmd}. Try /help.")
