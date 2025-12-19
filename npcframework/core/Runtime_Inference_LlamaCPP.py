from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from npcframework.core.NPC_Loader import load_npc
from npcframework.core.NPC_DB_Manager import NPCDatabase, Event
from npcframework.core.Runtime_Prompt_Compiler import compile_messages, CompileOptions, RuntimeInjection, ToolSpec
from npcframework.core.Runtime_Commands import handle_command
from npcframework.core.NPC_DB_Episodic_Promoter import EpisodicPromoter
from npcframework.core.Runtime_Orchestrator import run_with_tools, ToolRuntime

from npcframework.inference.llamacpp import LlamaCppEngine, LlamaCppConfig


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class RuntimeConfig:
    channel: str = "cli"
    environment_id: str = "local_cli"
    environment_name: str = "local_cli"

    # NOTE: These defaults are still here for CLI convenience,
    # but the APP can override via TurnRequest.recent_events_override
    default_history_limit: int = 20
    compile_history_limit: int = 20

    include_state_in_prompt: bool = True
    include_tools_in_prompt: bool = True

    debug_print_messages: bool = True
    debug_assert_messages_valid: bool = True

    model_path: str = "inference/models/gemma-3-4b-it-q4_0.gguf"
    model_n_ctx: int = 8192
    model_max_tokens: int = 256
    model_temperature: float = 0.7
    model_top_p: float = 0.9
    model_n_gpu_layers: int = 0

    default_state_mode: str = "idle"
    default_state_mood: str = "neutral"
    default_state_energy: float = 0.8

    runtime_goal: str = "help the user"
    runtime_mode: str = "conversational"
    runtime_perception_facts: List[str] = field(default_factory=list)
    runtime_env_facts: List[str] = field(default_factory=list)
    runtime_env_rules: List[str] = field(default_factory=list)
    runtime_additional_policies: List[str] = field(default_factory=list)
    identity_role_append: str = ""

    # Tooling (legacy defaults; APP should override via TurnRequest where possible)
    tool_promote_keywords: tuple[str, ...] = ()  # legacy heuristic fallback
    tool_builder: Optional[Callable[[], List[ToolSpec]]] = None
    tool_executor_builder: Optional[Callable[[], Dict[str, Callable[[Dict[str, Any]], Any]]]] = None


# =============================================================================
# REQUEST / RESULT
# =============================================================================

StreamCallback = Callable[[str], None]

@dataclass
class TurnRequest:
    npc_dir: str
    user_input: str

    stream_callback: Optional[StreamCallback] = None

    channel: Optional[str] = None
    environment_id: Optional[str] = None
    environment_name: Optional[str] = None

    perception_facts: Optional[List[str]] = None
    env_facts: Optional[List[str]] = None
    env_rules: Optional[List[str]] = None
    additional_policies: Optional[List[str]] = None
    identity_role_append: Optional[str] = None

    external_state: Optional[Dict[str, Any]] = None

    # ---- NEW: app-controlled turn inputs (drop-in; optional) ----
    allow_tool_calls: Optional[bool] = None
    tools_override: Optional[List[ToolSpec]] = None
    tool_handlers_override: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None

    # App can pre-filter events and bypass db.get_recent_events()
    recent_events_override: Optional[List[Event]] = None


@dataclass
class TurnResult:
    npc_name: str

    handled_by_system1: bool
    system1_response: Optional[str]
    should_exit: bool

    assistant_reply: Optional[str]
    wrote_user_event: bool
    wrote_assistant_event: bool

    compiled_messages: Optional[List[Dict[str, str]]] = None
    error: Optional[str] = None


# =============================================================================
# HELPERS
# =============================================================================

def _assert_messages_valid(messages: Any) -> None:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("invalid message shape")


def _ensure_default_state(db: NPCDatabase, cfg: RuntimeConfig) -> None:
    for k, v in {
        "mode": cfg.default_state_mode,
        "mood": cfg.default_state_mood,
        "energy": cfg.default_state_energy,
    }.items():
        if db.get_state(k) is None:
            db.set_state(k, v)


def _state_snapshot(db: NPCDatabase, cfg: RuntimeConfig) -> Dict[str, Any]:
    return {
        "mode": db.get_state("mode", cfg.default_state_mode),
        "mood": db.get_state("mood", cfg.default_state_mood),
        "energy": db.get_state("energy", cfg.default_state_energy),
    }


def _should_promote_tools_legacy(user_input: str, cfg: RuntimeConfig) -> bool:
    """
    Legacy v0 heuristic: promote tools if any keyword appears in user input.
    This exists only for backward compatibility.
    In microBB, you should NOT rely on this.
    """
    if not cfg.tool_promote_keywords:
        return False
    text = (user_input or "").lower()
    return any(k.lower() in text for k in cfg.tool_promote_keywords)

def _should_promote_tools(user_input: str, cfg: RuntimeConfig) -> bool:
    if not cfg.tool_builder:
        return False
    if not cfg.tool_promote_keywords:
        return False
    s = (user_input or "").lower()
    return any(k.lower() in s for k in cfg.tool_promote_keywords)


def _build_runtime_injection(
    user_input: str,
    state: Dict[str, Any],
    cfg: RuntimeConfig,
    req: TurnRequest
) -> RuntimeInjection:
    runtime_state = {
        "mode": cfg.runtime_mode,
        "goal": cfg.runtime_goal,
        "energy": state.get("energy"),
    }
    if req.external_state:
        runtime_state.update(req.external_state)

    # Tool availability boundary:
    # APP decides first. If not provided, fallback to legacy heuristic.
    if req.allow_tool_calls is not None:
        promote = bool(req.allow_tool_calls)
    else:
        promote = _should_promote_tools_legacy(user_input, cfg)

    # Tool list boundary:
    # APP can override the tool list per turn; otherwise fall back to cfg.tool_builder (if promoted)
    if promote:
        if req.tools_override is not None:
            tools = req.tools_override
        else:
            tools = cfg.tool_builder() if cfg.tool_builder else []
    else:
        tools = []

    return RuntimeInjection(
        environment_name=req.environment_name or cfg.environment_name,
        environment_facts=req.env_facts or cfg.runtime_env_facts,
        environment_rules=req.env_rules or cfg.runtime_env_rules,
        state=runtime_state,
        perception_facts=req.perception_facts or cfg.runtime_perception_facts,
        promote_tools=_should_promote_tools(user_input, cfg),
        available_tools=tools,

        additional_policies=req.additional_policies or cfg.runtime_additional_policies,
        identity_role_append=req.identity_role_append or cfg.identity_role_append,

    )


# =============================================================================
# ENGINE
# =============================================================================

def _resolve_path(p: str) -> str:
    """
    Resolves paths robustly:
    - If absolute, keep it.
    - If relative, resolve relative to project root (parent of /core).
    """
    path = Path(p)
    if path.is_absolute():
        return str(path)

    project_root = Path(__file__).resolve().parents[1]
    return str((project_root / path).resolve())


def build_engine(cfg: RuntimeConfig) -> LlamaCppEngine:
    model_path = _resolve_path(cfg.model_path)
    if not Path(model_path).exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    return LlamaCppEngine(
        LlamaCppConfig(
            model_path=model_path,
            n_ctx=cfg.model_n_ctx,
            max_tokens=cfg.model_max_tokens,
            temperature=cfg.model_temperature,
            top_p=cfg.model_top_p,
            n_gpu_layers=cfg.model_n_gpu_layers,
        )
    )


def warm_engine(engine: Any) -> None:
    """
    Forces model load + first-token path to run once.
    Helps avoid first-turn latency spikes in sims.
    """
    warm_messages = [
        {"role": "system", "content": "You are online. Reply with 'ready'."},
        {"role": "user", "content": "ping"},
    ]
    for _ in engine.chat_stream(warm_messages):
        pass


# =============================================================================
# CORE TURN RUNNER
# =============================================================================

def run_turn(*, req: TurnRequest, cfg: RuntimeConfig, engine: Any, npc=None, db=None) -> TurnResult:
    try:
        npc = npc or load_npc(req.npc_dir)
        npc_name = npc.manifest.get("display_name") or "NPC"

        db = db or NPCDatabase(npc.paths.db)
        db.init_db()
        _ensure_default_state(db, cfg)

        user_input = (req.user_input or "").strip()
        if not user_input:
            return TurnResult(npc_name, True, None, False, None, False, False)

        channel = req.channel or cfg.channel

        cmd = handle_command(user_input, npc=npc, npc_db=db)
        if cmd.handled:
            return TurnResult(npc_name, True, cmd.response, cmd.should_exit, None, False, False)

        # Always log user event
        db.add_event("user", user_input, meta={"channel": channel})

        # HISTORY boundary: app can override
        if req.recent_events_override is not None:
            recent = req.recent_events_override
        else:
            recent = db.get_recent_events(cfg.default_history_limit)

        state = _state_snapshot(db, cfg)
        runtime = _build_runtime_injection(user_input, state, cfg, req)

        messages = compile_messages(
            identity=npc.identity,
            persona=npc.persona,
            policy=npc.policy,
            recent_events=recent,
            runtime=runtime,
            options=CompileOptions(
                cfg.compile_history_limit,
                cfg.include_state_in_prompt,
                cfg.include_tools_in_prompt,
            ),
        )

        if cfg.debug_assert_messages_valid:
            _assert_messages_valid(messages)

        # TOOL boundary: app can override handlers per turn
        tools = runtime.available_tools if (runtime and runtime.promote_tools) else []
        if tools:
            if req.tool_handlers_override is not None:
                handlers = req.tool_handlers_override
            else:
                handlers = cfg.tool_executor_builder() if cfg.tool_executor_builder else {}
            schemas = {t.name: (t.schema or {}) for t in tools}
            tool_runtime = ToolRuntime(handlers=handlers, schemas=schemas)
        else:
            tool_runtime = None

        reply, _final_messages = run_with_tools(
            engine=engine,
            messages=messages,
            tool_runtime=tool_runtime,
            max_tool_steps=5,
            stream_callback=req.stream_callback,
            add_event=lambda role, content, meta: db.add_event(role, content, meta=meta),
            channel=channel,
        )

        db.add_event("assistant", reply, meta={"channel": channel})

        # Still default-promote (backward compatible).
        # microBB can replace this later with a hook.
        EpisodicPromoter(db).promote()

        return TurnResult(
            npc_name=npc_name,
            handled_by_system1=False,
            system1_response=None,
            should_exit=False,
            assistant_reply=reply,
            wrote_user_event=True,
            wrote_assistant_event=True,
            compiled_messages=messages,
        )

    except Exception as e:
        return TurnResult("NPC", False, None, False, None, False, False, error=str(e))


def run_cli(npc_dir: str):
    cfg = RuntimeConfig()
    engine = build_engine(cfg)
    warm_engine(engine)

    npc = load_npc(npc_dir)
    db = NPCDatabase(npc.paths.db)
    db.init_db()

    npc_name = npc.manifest.get("display_name") or "NPC"
    print(f"{npc_name} online.\n")

    while True:
        raw = input("You> ").strip()
        if not raw:
            continue

        print(f"{npc_name}> ", end="", flush=True)

        res = run_turn(
            req=TurnRequest(
                npc_dir=npc_dir,
                user_input=raw,
                stream_callback=lambda t: print(t, end="", flush=True),
            ),
            cfg=cfg,
            engine=engine,
            npc=npc,
            db=db,
        )

        print("\n")
        if res.should_exit:
            break


if __name__ == "__main__":
    import sys
    run_cli(sys.argv[1] if len(sys.argv) > 1 else "npc/kevin.npc")
