from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict, Callable
import time

from .npcframework_types import (
    EngineConfig,
    SessionConfig,
    TurnInput,
    TurnResult,
    TurnTrace,
    InferenceEngine,
    ToolHandlers,
)

# Inference backends
# NOTE: keep heavy deps (llama_cpp) optional by importing lazily.
from .inference.mock import MockEngine

# Core internals (package-safe)
from npcframework.core.NPC_Loader import load_npc
from npcframework.core.NPC_DB_Manager import NPCDatabase
from npcframework.core.Runtime_Prompt_Compiler import (
    RuntimeInjection,
    CompileOptions,
    compile_messages,
)
from npcframework.core.Runtime_Commands import handle_command
from npcframework.core.NPC_DB_Episodic_Promoter import EpisodicPromoter
from npcframework.core.Runtime_Orchestrator import run_with_tools, ToolRuntime

from .debug_dump import dump_messages

@dataclass
class RuntimeConfig:
    debug_dump_messages_json: bool = False
    debug_dump_messages_txt: bool = False
    debug_dump_dir: Optional[str] = None


# =============================================================================
# Engine
# =============================================================================

class Engine:
    """
    Long-lived inference container (keeps model hot).

    Owns ONLY the inference backend. No NPC knowledge.
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.backend: InferenceEngine = self._build_backend(cfg)

    def _build_backend(self, cfg: EngineConfig) -> InferenceEngine:
        if cfg.backend == "llamacpp":
            try:
                from .inference.llamacpp import LlamaCppEngine, LlamaCppConfig  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "llamacpp backend requires llama-cpp-python. Install it, or use backend='mock'."
                ) from e
            llama_cfg = LlamaCppConfig(
                model_path=cfg.model_path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.n_threads,
                n_gpu_layers=cfg.n_gpu_layers,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                stop=cfg.stop,
            )
            return LlamaCppEngine(llama_cfg)

        if cfg.backend == "mock":
            return MockEngine()

        raise ValueError(f"Unknown backend: {cfg.backend}")

    def warmup(self) -> None:
        """Force model load + first token path."""
        warm = [
            {"role": "system", "content": "You are online. Reply with 'ready'."},
            {"role": "user", "content": "ping"},
        ]
        for _ in self.backend.chat_stream(warm):
            pass


# =============================================================================
# Session
# =============================================================================

class Session:
    """
    Entity-bound runner: NPC identity/persona/policy + DB.
    Reusable across turns. Library-safe.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        npc_dir: str,
        cfg: Optional[SessionConfig] = None,
        db: Optional[NPCDatabase] = None,
        runtime_cfg: Optional[RuntimeConfig] = None,   # ✅ NEW
    ) -> None:
        self.engine = engine
        self.cfg = cfg or SessionConfig()

        # ✅ NEW: runtime_cfg overrides SessionConfig for debug dumping only
        if runtime_cfg is not None:
            if runtime_cfg.debug_dump_messages_json is not None:
                self.cfg.debug_dump_messages_json = runtime_cfg.debug_dump_messages_json
            if runtime_cfg.debug_dump_messages_txt is not None:
                self.cfg.debug_dump_messages_txt = runtime_cfg.debug_dump_messages_txt
            if runtime_cfg.debug_dump_dir is not None:
                self.cfg.debug_dump_dir = runtime_cfg.debug_dump_dir

        # --- Load NPC
        self.npc = load_npc(npc_dir)
        self.npc_name = (
            self.npc.manifest.get("display_name")
            or self.npc.manifest.get("name")
            or "NPC"
        )

        # --- DB is NOT optional in practice
        self.db = db or NPCDatabase(self.npc.paths.db)
        self.db.init_db()   # <- guarantees file exists

        self._ensure_default_state()

    # -----------------------
    # State helpers
    # -----------------------

    def _ensure_default_state(self) -> None:
        defaults = {
            "mode": self.cfg.default_state_mode,
            "mood": self.cfg.default_state_mood,
            "energy": self.cfg.default_state_energy,
        }
        for k, v in defaults.items():
            if self.db.get_state(k) is None:
                self.db.set_state(k, v)

    def _state_snapshot(self) -> Dict[str, Any]:
        return {
            "mode": self.db.get_state("mode", self.cfg.default_state_mode),
            "mood": self.db.get_state("mood", self.cfg.default_state_mood),
            "energy": self.db.get_state("energy", self.cfg.default_state_energy),
        }

    # -----------------------
    # Turn runner
    # -----------------------

    def _compile_messages(self, *, inj: RuntimeInjection) -> list[dict[str, str]]:
        """Compile the message list that will be passed to the inference backend."""
        recent = self.db.get_recent_events(self.cfg.history_limit)
        return compile_messages(
            identity=self.npc.identity,
            persona=self.npc.persona,
            policy=self.npc.policy,
            recent_events=recent,
            runtime=inj,
            options=CompileOptions(
                history_limit=self.cfg.history_limit,
                include_state=self.cfg.include_state_in_prompt,
                include_tools=self.cfg.include_tools_in_prompt,
            ),
        )

    def run_turn(self, turn: TurnInput) -> TurnResult:
        """
        Canonical library entrypoint.
        ALWAYS:
        - initializes DB
        - logs user input
        - returns TurnResult (never raises)
        """
        trace = TurnTrace()

        try:
            user_input = (turn.user_input or "").strip()
            if not user_input:
                return TurnResult(
                    npc_name=self.npc_name,
                    assistant_reply=None,
                    trace=trace,
                )

            # -----------------------
            # System-1 (commands)
            # -----------------------
            cmd = handle_command(user_input, npc=self.npc, npc_db=self.db)
            if cmd.handled:
                if cmd.response:
                    self.db.add_event("assistant", cmd.response, meta={"system": "command"})
                return TurnResult(
                    npc_name=self.npc_name,
                    assistant_reply=cmd.response,
                    handled_by_system1=True,
                    system1_response=cmd.response,
                    should_exit=cmd.should_exit,
                    trace=trace,
                )

            # -----------------------
            # DB truth FIRST
            # -----------------------
            self.db.add_event("user", user_input, meta={"channel": self.cfg.channel})

            # -----------------------
            # Runtime injection
            # -----------------------
            state = self._state_snapshot()
            runtime_state = {
                "mode": self.cfg.runtime_mode,
                "goal": self.cfg.runtime_goal,
                "energy": state["energy"],
            }
            if turn.external_state:
                runtime_state.update(turn.external_state)

            inj = RuntimeInjection(
                environment_name=turn.environment_name or self.cfg.environment_name,
                environment_facts=turn.environment_facts or self.cfg.runtime_env_facts,
                environment_rules=turn.environment_rules or self.cfg.runtime_env_rules,
                perception_facts=turn.perception_facts or self.cfg.runtime_perception_facts,
                state=runtime_state,
                additional_policies=turn.additional_policies or self.cfg.runtime_additional_policies,
                identity_role_append=turn.identity_role_append or self.cfg.identity_role_append,
                available_tools=turn.available_tools or [],
                promote_tools=bool(turn.available_tools),
            )

            # -----------------------
            # Prompt compilation
            # -----------------------
            messages = self._compile_messages(inj=inj)
            trace.compiled_messages = messages

            if self.cfg.debug_assert_messages_valid:
                _assert_messages_valid(messages)

            # -----------------------
            # Debug dumps (what goes into the model)
            # -----------------------
            if self.cfg.debug_dump_messages_json or self.cfg.debug_dump_messages_txt:
                turn_id = str(int(time.time() * 1000))
                dump_messages(
                    messages=messages,
                    out_dir=self.cfg.debug_dump_dir,
                    turn_id=turn_id,
                    write_json=self.cfg.debug_dump_messages_json,
                    write_txt=self.cfg.debug_dump_messages_txt,
                    meta={
                        "npc": self.npc_name,
                        "channel": self.cfg.channel,
                        "history_limit": self.cfg.history_limit,
                        "include_state": self.cfg.include_state_in_prompt,
                        "include_tools": self.cfg.include_tools_in_prompt,
                    },
                )

            # -----------------------
            # Tool runtime (caller-owned)
            # -----------------------
            handlers: ToolHandlers = turn.tool_handlers or {}
            schemas = {t.name: (t.schema or {}) for t in (inj.available_tools or [])}
            tool_runtime = ToolRuntime(
                handlers=handlers,
                schemas=schemas,
            ) if schemas else None

            # -----------------------
            # Orchestration
            # -----------------------
            reply, _ = run_with_tools(
                engine=self.engine.backend,
                messages=messages,
                tool_runtime=tool_runtime,
                max_tool_steps=5,
                stream_callback=turn.stream_callback,
                add_event=lambda role, content, meta=None: self.db.add_event(role, content, meta=meta),
                channel=self.cfg.channel,
                user_input=turn.user_input,  # ✅ FIXED
            )

            # -----------------------
            # Persist assistant + memory
            # -----------------------
            self.db.add_event("assistant", reply, meta={"channel": self.cfg.channel})
            EpisodicPromoter(self.db).promote()

            return TurnResult(
                npc_name=self.npc_name,
                assistant_reply=reply,
                trace=trace,
            )

        except Exception as e:
            return TurnResult(
                npc_name=self.npc_name,
                assistant_reply=None,
                trace=trace,
                error=str(e),
            )

# =============================================================================
# Debug helpers
# =============================================================================

def _assert_messages_valid(messages: Any) -> None:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("invalid message shape")
