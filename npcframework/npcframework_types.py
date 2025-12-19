from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Iterable, Tuple


# Reuse your ToolSpec if you want; this keeps the API stable even if ToolSpec moves later.
try:
    from npcframework.core.Runtime_Prompt_Compiler import ToolSpec  # type: ignore
except Exception:
    ToolSpec = Any  # fallback for type-checkers / packaging

Message = Dict[str, str]
StreamCallback = Callable[[str], None]
ToolHandler = Callable[[Dict[str, Any]], Any]
ToolHandlers = Dict[str, ToolHandler]


class InferenceEngine(Protocol):
    """
    Minimal contract NPCFramework expects from an inference backend.

    Your LlamaCppEngine already matches this.
    """
    def chat_stream(self, messages: List[Message]) -> Iterable[str]: ...
    def chat(self, messages: List[Message]) -> str: ...

ToolValidator = Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str], Optional[Dict[str, Any]]]]

@dataclass
class ToolRuntime:
    """
    Runtime tool binding for a turn/session.
    - schemas: tool name -> schema dict (json-schema-ish)
    - handlers: tool name -> callable(args)->any
    - validate_call: optional trust boundary: approve/deny/patch tool calls
    """
    handlers: ToolHandlers = field(default_factory=dict)
    schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validate_call: Optional[ToolValidator] = None

@dataclass
class EngineConfig:
    """
    Inference-only configuration.

    If you want to keep your existing RuntimeConfig, that's fine—this is the public API surface.
    Internally you can map EngineConfig -> RuntimeConfig or LlamaCppConfig.
    """
    backend: str = "llamacpp"

    model_path: str = ""
    n_ctx: int = 8192
    n_threads: Optional[int] = None
    n_gpu_layers: int = 0

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256

    stop: Optional[List[str]] = None


@dataclass
class SessionConfig:
    """
    Session / compilation / runtime settings.

    This is *not* world logic. It's how the NPC is run.
    """
    channel: str = "cli"
    environment_id: str = "local"
    environment_name: str = "local"

    # prompt/history
    history_limit: int = 20
    include_state_in_prompt: bool = True
    include_tools_in_prompt: bool = True

    # debugging (library should not print, but caller can opt into returning debug data)
    debug_assert_messages_valid: bool = True

    # default NPC state seeding
    default_state_mode: str = "idle"
    default_state_mood: str = "neutral"
    default_state_energy: float = 0.8

    # “runtime state” fields (optional defaults)
    runtime_goal: str = "help the user"
    runtime_mode: str = "conversational"

    runtime_env_facts: List[str] = field(default_factory=list)
    runtime_env_rules: List[str] = field(default_factory=list)
    runtime_perception_facts: List[str] = field(default_factory=list)

    runtime_additional_policies: List[str] = field(default_factory=list)
    identity_role_append: str = ""


@dataclass
class TurnInput:
    """
    Fully parameterized input for a single NPC turn.

    Tool trust boundary:
    - The caller decides what tools + handlers to provide.
    - Passing [] / {} means "no tools this turn".
    """
    user_input: str

    # runtime injection (caller overrides; if None, SessionConfig defaults apply)
    environment_name: Optional[str] = None
    environment_facts: Optional[List[str]] = None
    environment_rules: Optional[List[str]] = None
    perception_facts: Optional[List[str]] = None

    # state injection (merged with session snapshot; app/world owns truth here)
    external_state: Optional[Dict[str, Any]] = None

    # policy/identity extensions (caller controlled)
    additional_policies: Optional[List[str]] = None
    identity_role_append: Optional[str] = None

    # tools (caller controlled: trust boundary lives here)
    available_tools: Optional[List[ToolSpec]] = None
    tool_handlers: Optional[ToolHandlers] = None

    # streaming
    stream_callback: Optional[StreamCallback] = None


@dataclass
class ToolCallTrace:
    tool: str
    args: Dict[str, Any]
    raw_line: str


@dataclass
class ToolResultTrace:
    tool: str
    ok: bool
    result: Any = None
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    raw_line: Optional[str] = None


@dataclass
class TurnTrace:
    """
    Structured outputs for observability + debugging.
    """
    compiled_messages: Optional[List[Message]] = None
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    tool_results: List[ToolResultTrace] = field(default_factory=list)


@dataclass
class TurnResult:
    """
    What the library returns. No prints, no side effects beyond DB writes inside Session.

    The caller decides rendering.
    """
    npc_name: str
    assistant_reply: Optional[str]

    should_exit: bool = False
    handled_by_system1: bool = False
    system1_response: Optional[str] = None

    trace: TurnTrace = field(default_factory=TurnTrace)

    error: Optional[str] = None
