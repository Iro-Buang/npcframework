from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict
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

from .inference.mock import MockEngine

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


class Engine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.backend: InferenceEngine = self._build_backend(cfg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Engine":
        """Create an Engine from a canonical YAML config (single source of truth)."""
        from npcframework.inference.factory import engine_config_from_yaml
        cfg = engine_config_from_yaml(path)
        return cls(cfg)

    @classmethod
    def from_config_dir(cls, config_dir: str | Path, filename: str = "inference.yaml") -> "Engine":
        p = Path(config_dir) / filename
        return cls.from_yaml(p)

    def _build_backend(self, cfg: EngineConfig) -> InferenceEngine:
        if cfg.backend == "llamacpp":
            try:
                from .inference.llamacpp import LlamaCppEngine, LlamaCppConfig  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "llamacpp backend requires optional dependency.\n"
                    "Install with: pip install npcframework[llamacpp]\n"
                    "or use backend='mock'."
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

        if cfg.backend == "vllm":
            from .inference.vllm import VLLMEngine, VLLMConfig

            vcfg = VLLMConfig(
                base_url=cfg.vllm_base_url,
                model=cfg.vllm_model,
                api_key=cfg.vllm_api_key,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                timeout_s=cfg.vllm_timeout_s,
                stream=cfg.vllm_stream,
            )
            return VLLMEngine(vcfg)

        if cfg.backend == "mock":
            return MockEngine()

        raise ValueError(f"Unknown backend: {cfg.backend}")

    def warmup(self) -> None:
        warm = [
            {"role": "system", "content": "You are online. Reply with 'ready'."},
            {"role": "user", "content": "ping"},
        ]
        for _ in self.backend.chat_stream(warm):
            pass



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
        runtime_cfg: Optional[RuntimeConfig] = None,

        # --- event DB scoping knobs (config-first, but overridable here) ---
        channel: Optional[str] = None,
        environment_id: Optional[str] = None,
        environment_name: Optional[str] = None,
        session_id: Optional[str] = None,
        run_across_sessions: Optional[bool] = None,
        run_across_channels: Optional[bool] = None,
        run_across_environments: Optional[bool] = None,
    ) -> None:
        self.engine = engine
        self.cfg = cfg or SessionConfig()
        # Apply explicit overrides (if provided)
        if channel is not None:
            self.cfg.channel = str(channel)
        if environment_id is not None:
            self.cfg.environment_id = str(environment_id)
        if environment_name is not None:
            self.cfg.environment_name = str(environment_name)
        if run_across_sessions is not None:
            self.cfg.run_across_sessions = bool(run_across_sessions)
        if run_across_channels is not None:
            self.cfg.run_across_channels = bool(run_across_channels)
        if run_across_environments is not None:
            self.cfg.run_across_environments = bool(run_across_environments)

        # Session scoping is the default behavior. If caller did not provide a session_id,
        # generate a new one per Session instance so history stays isolated by default.
        if session_id is not None:
            self.cfg.session_id = str(session_id)
        elif self.cfg.session_id is None:
            import uuid as _uuid_mod
            self.cfg.session_id = f"sess_{_uuid_mod.uuid4().hex}"

        # runtime_cfg overrides SessionConfig for debug dumping only
        if runtime_cfg is not None:
            if runtime_cfg.debug_dump_messages_json is not None:
                self.cfg.debug_dump_messages_json = runtime_cfg.debug_dump_messages_json
            if runtime_cfg.debug_dump_messages_txt is not None:
                self.cfg.debug_dump_messages_txt = runtime_cfg.debug_dump_messages_txt
            if runtime_cfg.debug_dump_dir is not None:
                self.cfg.debug_dump_dir = runtime_cfg.debug_dump_dir

        # Load NPC
        self.npc = load_npc(npc_dir)
        self.npc_name = (
            self.npc.manifest.get("display_name")
            or self.npc.manifest.get("name")
            or "NPC"
        )

        # DB (not optional in practice)
        self.db = db or NPCDatabase(
            self.npc.paths.db,
            npc_id=str(self.npc.manifest.get("id") or "npc"),
            default_channel=self.cfg.channel,
            default_environment_id=self.cfg.environment_id,
            run_across_sessions=self.cfg.run_across_sessions,
            run_across_environments=self.cfg.run_across_environments,
            run_across_channels=self.cfg.run_across_channels,
            session_id=self.cfg.session_id,
        )
        self.db.init_db()

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
    # Prompt compilation
    # -----------------------

    def _resolve_include_tools(self, turn: TurnInput) -> bool:
        """
        Per-turn tool prompt toggle:
        - If TurnInput.include_tools_in_prompt is set -> use it
        - Else -> use SessionConfig.include_tools_in_prompt default
        """
        include_tools = self.cfg.include_tools_in_prompt
        if getattr(turn, "include_tools_in_prompt", None) is not None:
            include_tools = bool(turn.include_tools_in_prompt)
        return include_tools

    def _compile_messages(self, *, inj: RuntimeInjection, turn: TurnInput) -> list[dict[str, str]]:
        """
        Compile the message list that will be passed to the inference backend.
        """
        recent = self.db.get_recent_events(
            self.cfg.history_limit,
            session_id=self.cfg.session_id,
            channel=self.cfg.channel,
            environment_id=self.cfg.environment_id,
            run_across_sessions=self.cfg.run_across_sessions,
            run_across_channels=self.cfg.run_across_channels,
            run_across_environments=self.cfg.run_across_environments,
        )
        include_tools = self._resolve_include_tools(turn)

        return compile_messages(
            identity=self.npc.identity,
            persona=self.npc.persona,
            policy=self.npc.policy,
            recent_events=recent,
            runtime=inj,
            options=CompileOptions(
                history_limit=self.cfg.history_limit,
                include_state=self.cfg.include_state_in_prompt,
                include_perception=self.cfg.include_perception_in_prompt,
                include_memory=self.cfg.include_memory_in_prompt,
                include_tools=include_tools,
            ),
        )

    # -----------------------
    # Turn runner
    # -----------------------

    def run_turn(self, turn: TurnInput) -> TurnResult:
        """
        Canonical library entrypoint.
        ALWAYS:
        - initializes DB
        - logs user input
        - returns TurnResult (never raises)
        """
        turn_id = str(int(time.time() * 1000))
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
            self.db.add_event("user", user_input, meta={"channel": self.cfg.channel, "turn_id": turn_id})

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

            # ---- GOALS: existential (npc yaml) + transient (runtime injected) ----
            npc_goals = getattr(self.npc, "goals", None) or {}
            existential = npc_goals.get("existential") or []

            existential_goals: list[str] = []
            if isinstance(existential, list):
                for item in existential:
                    if isinstance(item, str) and item.strip():
                        existential_goals.append(item.strip())
                    elif isinstance(item, dict):
                        t = item.get("text", "")
                        if isinstance(t, str) and t.strip():
                            existential_goals.append(t.strip())

            transient_goals = turn.transient_goals or []
            if not isinstance(transient_goals, list):
                transient_goals = [str(transient_goals)]

            # Tools: resolve visibility + style once
            include_tools = self._resolve_include_tools(turn)
            available_tools = turn.available_tools or []
            requested_style = getattr(turn, "tool_prompt_style", None) or "compact"
            tool_prompt_style = (requested_style if include_tools else "none")

            inj = RuntimeInjection(
                environment_name=turn.environment_name or self.cfg.environment_name,
                environment_facts=turn.environment_facts or self.cfg.runtime_env_facts,
                environment_rules=turn.environment_rules or self.cfg.runtime_env_rules,
                perception_facts=turn.perception_facts or self.cfg.runtime_perception_facts,
                existential_goals=existential_goals,
                transient_goals=transient_goals,
                working_memory=turn.working_memory or [],
                recalled_contexts=turn.recalled_contexts or [],
                semantic_memory=turn.semantic_memory or [],
                state=runtime_state,
                additional_policies=turn.additional_policies or self.cfg.runtime_additional_policies,
                identity_role_append=turn.identity_role_append or self.cfg.identity_role_append,
                # ✅ Tools: consistent wiring
                available_tools=(available_tools if include_tools else []),
                promote_tools=bool(include_tools and available_tools),
                tool_prompt_style=tool_prompt_style,
            )

            # -----------------------
            # Prompt compilation
            # -----------------------
            messages = self._compile_messages(inj=inj, turn=turn)
            trace.compiled_messages = messages

            if self.cfg.debug_assert_messages_valid:
                _assert_messages_valid(messages)

            # -----------------------
            # Debug dumps (what goes into the model)
            # -----------------------
            if self.cfg.debug_dump_messages_json or self.cfg.debug_dump_messages_txt:
                dump_messages(
                    messages=messages,
                    out_dir=self.cfg.debug_dump_dir,
                    turn_id=turn_id,
                    write_json=self.cfg.debug_dump_messages_json,
                    write_txt=self.cfg.debug_dump_messages_txt,
                    meta={
                        "npc": self.npc_name,
                        "channel": self.cfg.channel,
                        "turn_id": turn_id,
                        "history_limit": self.cfg.history_limit,
                        "include_state": self.cfg.include_state_in_prompt,
                        "include_tools": include_tools,
                        "tool_prompt_style": tool_prompt_style,
                        "allow_spontaneous_tools": getattr(turn, "allow_spontaneous_tools", False),
                    },
                )

            # -----------------------
            # Tool runtime (caller-owned)
            # -----------------------
            handlers: ToolHandlers = turn.tool_handlers or {}
            schemas = {t.name: (t.schema or {}) for t in (inj.available_tools or [])}

            tool_runtime = (
                ToolRuntime(handlers=handlers, schemas=schemas)
                if include_tools and schemas
                else None
            )

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
                user_input=user_input,
                allow_spontaneous_tools=getattr(turn, "allow_spontaneous_tools", False),
                turn_id=turn_id,  # ✅ correlate logs/dumps/tool events
            )

            # -----------------------
            # Persist assistant + memory
            # -----------------------
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


def _assert_messages_valid(messages: Any) -> None:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("invalid message shape")
