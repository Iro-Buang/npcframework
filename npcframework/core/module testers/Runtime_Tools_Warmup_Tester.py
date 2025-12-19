from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Callable, Any, List

from npcframework.core.Runtime_Inference_LlamaCPP import (
    RuntimeConfig,
    TurnRequest,
    build_engine,
    warm_engine,
    run_turn,
)

from npcframework.core.Runtime_Prompt_Compiler import ToolSpec
from npcframework.core.NPC_Loader import load_npc
from npcframework.core.NPC_DB_Manager import NPCDatabase


# =============================================================================
# TUNABLES (touch grass everywhere else)
# =============================================================================

# NPC
NPC_REL_DIR = Path("npc") / "kevin.npc"

# Tool promotion
TOOL_PROMOTE_KEYWORDS = ("tool", "add", "calculate", "sum")

# Test turns
TURN_1_TEXT = "What's a Kevin?'"
TURN_2_TEXT = "Do you remember what what we just talked about?"

# Output controls
PRINT_RESOLVED_PATHS = False
PRINT_TURN_STATUS = True

# Tool trace controls (reads from NPC sqlite event log)
SHOW_TOOL_TRACE = True
TOOL_TRACE_TAIL_N = 30               # how many recent events to scan
TOOL_TRACE_ONLY_SYSTEM = True        # only show system role tool events
TOOL_TRACE_MATCH = ("/tool_call", "/tool_result")  # strings to match

# Root discovery
PROJECT_ROOT_MAX_UP = 12

# Runtime config overrides for this test
RUNTIME_CFG_OVERRIDES = dict(
    channel="test",
    environment_id="test_env",
    environment_name="unit_test",
    debug_print_messages=False,
    debug_assert_messages_valid=True,
)


# =============================================================================
# Tool wiring (APP LAYER)
# =============================================================================

def tool_specs():
    return [
        ToolSpec(
            name="add",
            description="Adds two numbers. Use when user asks to add/calculate.",
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            few_shots=[
                {"input": "Add 2 and 5", "output": '/tool_call add {"a": 2, "b": 5}'}
            ],
        )
    ]


def tool_handlers() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    def add(args: Dict[str, Any]):
        return {"sum": args["a"] + args["b"]}

    return {"add": add}


# =============================================================================
# Plumbing
# =============================================================================

def find_project_root(start: Path, *, max_up: int) -> Path:
    """
    Finds repo root by walking upward until both 'core/' and 'npc/' exist.
    This keeps tests runnable from any working directory.
    """
    cur = start.resolve()
    for _ in range(max_up):
        if (cur / "core").exists() and (cur / "npc").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(
        f"Could not find project root from {start} after {max_up} steps "
        "(expected 'core/' and 'npc/' dirs)."
    )


def resolve_npc_dir() -> str:
    root = find_project_root(Path(__file__).parent, max_up=PROJECT_ROOT_MAX_UP)
    npc_dir = (root / NPC_REL_DIR).resolve()
    if PRINT_RESOLVED_PATHS:
        print("[project_root]", root)
        print("[npc_dir]", npc_dir)
    return str(npc_dir)


def open_db(npc_dir: str) -> NPCDatabase:
    npc = load_npc(npc_dir)
    db = NPCDatabase(npc.paths.db)
    db.init_db()
    return db


def print_tool_trace(db: NPCDatabase, *, tail_n: int) -> None:
    if not SHOW_TOOL_TRACE:
        return

    events = db.get_recent_events(tail_n)

    lines: List[str] = []
    for e in events:
        # Defensive: Event shape depends on your DB manager implementation.
        role = getattr(e, "role", None)
        content = getattr(e, "content", "") or ""

        if TOOL_TRACE_ONLY_SYSTEM and role != "system":
            continue

        if any(tok in content for tok in TOOL_TRACE_MATCH):
            lines.append(f"- {content}")

    if not lines:
        print("[tool_trace] (none)")
        return

    print("[tool_trace]")
    for line in lines:
        print(" ", line)


def run_and_print(cfg: RuntimeConfig, engine: Any, npc_dir: str, db: NPCDatabase, user_text: str) -> None:
    print("\n" + "=" * 80)
    print("You> " + user_text)
    print("NPC> ", end="", flush=True)

    # print("[dbg] debug_print_messages =", cfg.debug_print_messages)

    res = run_turn(
        req=TurnRequest(
            npc_dir=npc_dir,
            user_input=user_text,
            stream_callback=lambda t: print(t, end="", flush=True),
        ),
        cfg=cfg,
        engine=engine,
    )

    print("\n" + "-" * 80)
    if res.error:
        print("[error]", res.error)
        return

    if SHOW_TOOL_TRACE:
        print_tool_trace(db, tail_n=TOOL_TRACE_TAIL_N)

    if PRINT_TURN_STATUS:
        print("[ok] handled_by_system1:", res.handled_by_system1)
        print("[ok] reply_len:", len(res.assistant_reply or ""))
        print("[ok] tool_promotion_keywords:", cfg.tool_promote_keywords)
        print("[ok] should_exit:", res.should_exit)


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = RuntimeConfig(**RUNTIME_CFG_OVERRIDES)

    # Enable tools + promotion
    cfg.tool_builder = tool_specs
    cfg.tool_executor_builder = tool_handlers
    cfg.tool_promote_keywords = TOOL_PROMOTE_KEYWORDS

    engine = build_engine(cfg)

    # Warmup timing
    t0 = time.time()
    warm_engine(engine)
    t1 = time.time()
    print(f"[warmup] done in {int((t1 - t0) * 1000)} ms")

    npc_dir = resolve_npc_dir()
    db = open_db(npc_dir)

    # Turn 1: normal
    run_and_print(cfg, engine, npc_dir, db, TURN_1_TEXT)

    # Turn 2: should tool-call
    run_and_print(cfg, engine, npc_dir, db, TURN_2_TEXT)


if __name__ == "__main__":
    main()
