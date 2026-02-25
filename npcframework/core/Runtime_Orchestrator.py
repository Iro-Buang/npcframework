from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .Runtime_Config import load_config_yaml, get_str

Message = Dict[str, str]
ToolHandler = Callable[[Dict[str, Any]], Any]
ToolValidator = Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str], Optional[Dict[str, Any]]]]

cfg = load_config_yaml("runtime_orchestrator")

TOOL_CALL_PREFIX = get_str(cfg, "tool_call_prefix", "/tool_call", filename="runtime_orchestrator")
TOOL_RESULT_PREFIX = get_str(cfg, "tool_result_prefix", "/tool_result", filename="runtime_orchestrator")

FINALIZE_SYSTEM_PROMPT = get_str(
    cfg,
    "finalize_system_prompt",
    """Tool execution is complete. Now answer the user's request.
Rules:
- Do NOT output /tool_call
- Do NOT output JSON
- Do NOT explain your reasoning
- Output ONLY the final answer requested by the user
""",
)

@dataclass
class ToolRuntime:
    handlers: Dict[str, ToolHandler]
    schemas: Dict[str, Dict[str, Any]]
    validate_call: Optional[ToolValidator] = None


def parse_tool_call(text: str, *, relaxed: bool = False) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Robust tool-call parser.

    Accepts:
      1) /tool_call {"name":"time_now","args":{...}}
      2) /tool_call {"tool":"time_now","args":{...}}
      3) /tool_call time_now
      4) /tool_call get_time
      5) /tool_call time_now {"a":1,"b":2}
      6) /tool_call time_now {"args":{...}}

    Normalizes into: (tool_name, args_dict)
    """
    if not text:
        return None

    s = text.strip()

    # Strict mode: tool call must be the entire assistant output line
    if not s.startswith(TOOL_CALL_PREFIX):
        if not relaxed:
            return None

        # Relaxed mode: scan for a tool_call line somewhere inside
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        hit = next((ln for ln in lines if ln.startswith(TOOL_CALL_PREFIX)), None)
        if not hit:
            return None
        s = hit

    payload = s[len(TOOL_CALL_PREFIX):].strip()
    if not payload:
        raise ValueError("tool_call missing payload")

    cfg_aliases = load_config_yaml("runtime_orchestrator").get("tool_aliases") or {}
    if not isinstance(cfg_aliases, dict):
        cfg_aliases = {}
    TOOL_ALIASES = {
        "get_time": "time_now",
        "time": "time_now",
        "now": "time_now",
        **{str(k): str(v) for k, v in cfg_aliases.items()},
    }

    def _apply_alias(name: str) -> str:
        n = (name or "").strip()
        if not n:
            return n
        return TOOL_ALIASES.get(n, n)

    decoder = json.JSONDecoder()

    if payload.startswith("{"):
        try:
            obj, _idx = decoder.raw_decode(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"tool_call json invalid: {e}") from e

        if not isinstance(obj, dict):
            raise ValueError("tool_call payload must be a JSON object (dict)")

        name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool_call json missing tool name (expected name)")

        args = obj.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError("tool_call args must be a JSON object (dict)")

        return _apply_alias(name), args

    parts = payload.split(maxsplit=1)
    tool_name = _apply_alias(parts[0])
    if not tool_name:
        raise ValueError("tool_call tool_name is empty")

    if len(parts) == 1:
        return tool_name, {}

    rest = parts[1].strip()
    if not rest:
        return tool_name, {}

    if rest.startswith("{"):
        try:
            obj, _idx = decoder.raw_decode(rest)
        except json.JSONDecodeError:
            return tool_name, {}

        if isinstance(obj, dict):
            if "args" in obj and isinstance(obj.get("args"), dict):
                return tool_name, obj["args"]
            return tool_name, obj

    return tool_name, {}

def _try_autorun_tool_from_user_input(user_input: str, tool_runtime: Optional[ToolRuntime]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Minimal deterministic autorun for obvious tool requests.
    Returns (tool_name, args) or None.
    """
    if not tool_runtime or not tool_runtime.handlers:
        return None

    u = (user_input or "").lower().strip()

    autorun_cfg = load_config_yaml("runtime_orchestrator").get("autorun") or {}
    if not isinstance(autorun_cfg, dict):
        autorun_cfg = {}

    def _has_any_keyword(tool: str) -> bool:
        rule = autorun_cfg.get(tool) or {}
        if not isinstance(rule, dict):
            return False
        kws = rule.get("keywords") or []
        if not isinstance(kws, list):
            return False
        return any(str(k).lower() in u for k in kws if str(k).strip())

    # If the user asked for time, and time_now exists
    if "time_now" in tool_runtime.handlers and _has_any_keyword("time_now"):
        return ("time_now", {})

    # If the user asked to add and 'add' tool exists, try parse "A + B"
    if "add" in tool_runtime.handlers and _has_any_keyword("add"):
        import re
        m = re.search(r"(-?\d+)\s*\+\s*(-?\d+)", u)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            return ("add", {"a": a, "b": b})

    return None


def _user_demands_tool(user_input: str) -> bool:
    u = (user_input or "").lower()
    cfg_triggers = load_config_yaml("runtime_orchestrator").get("user_demands_tool_triggers") or []
    if not isinstance(cfg_triggers, list):
        cfg_triggers = []
    triggers = [str(t).lower() for t in cfg_triggers if str(t).strip()]
    if not triggers:
        triggers = [
            "use a tool",
            "use tool",
            "call a tool",
            "tool call",
            "use your tool",
            "use the tool",
        ]
    return any(t in u for t in triggers)


def format_tool_result(tool_name: str, *, ok: bool, result: Any = None, error: Optional[str] = None, latency_ms: Optional[int] = None) -> str:
    payload: Dict[str, Any] = {
        "ok": ok,
        "tool": tool_name,
        "result": result,
        "error": error,
        "latency_ms": latency_ms,
    }
    return f"{TOOL_RESULT_PREFIX} {tool_name} {json.dumps(payload, ensure_ascii=False)}"

def extract_canonical_answer(result: Any) -> Optional[str]:
    """
    Canonical extraction (framework-level contract):
    - If result is dict and contains 'answer' scalar -> that's the canonical.
    - Else fallback: scalar -> canonical
    - Else fallback: dict with a single scalar leaf -> canonical
    """
    if result is None:
        return None

    # 1) Preferred contract: {"answer": <scalar>, ...}
    if isinstance(result, dict) and "answer" in result:
        ans = result.get("answer")
        if isinstance(ans, (str, int, float, bool)) and not isinstance(ans, bool) or isinstance(ans, bool):
            return str(ans)

    # 2) Scalar
    if isinstance(result, (str, int, float, bool)):
        return str(result)

    # 3) Old fallback: exactly one scalar leaf in dict
    if isinstance(result, dict):
        leaves: List[Any] = []

        def walk(v):
            if isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)
            else:
                leaves.append(v)

        walk(result)

        if len(leaves) == 1 and isinstance(leaves[0], (str, int, float, bool)):
            return str(leaves[0])

    return None



def _type_matches(value: Any, type_name: str) -> bool:
    t = (type_name or "").lower()
    if t == "string":
        return isinstance(value, str)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    return True


def validate_args(schema: Optional[Dict[str, Any]], args: Dict[str, Any]) -> None:
    if not schema:
        return
    required = schema.get("required") or []
    for k in required:
        if k not in args:
            raise ValueError(f"missing required field: {k}")
    props = schema.get("properties") or {}
    for k, spec in props.items():
        if k not in args:
            continue
        if not isinstance(spec, dict):
            continue
        t = spec.get("type")
        if t and not _type_matches(args[k], t):
            raise ValueError(f"field {k!r} must be {t}")


def run_with_tools(
    *,
    engine: Any,
    messages: List[Message],
    tool_runtime: Optional[ToolRuntime],
    max_tool_steps: int = 5,
    stream_callback: Optional[Callable[[str], None]] = None,
    add_event: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    channel: str = "runtime",
    user_input: str = "",
    turn_id: Optional[str] = None,
    # ✅ NEW: prevent DB duplicate writes by letting Session own assistant final logging
    log_final_answer_event: bool = True,
    # ✅ NEW: if True, after a successful tool call we refuse any further tool calls and force finalize
    hard_stop_after_successful_tool: bool = True,
    # ✅ NEW: allow tools even when user did NOT ask (simulation/agent mode)
    allow_spontaneous_tools: bool = False,
) -> Tuple[str, List[Message]]:
    """Run inference with tool-call support + deterministic autorun + canonical answer enforcement.

    Behavioral gates:
    - Tools are allowed if (user demanded tool) OR (allow_spontaneous_tools=True).
    - Autorun heuristic triggers ONLY when user demanded a tool.
    - If model tool-calls when tools are NOT allowed, we block and force a normal answer.
    - If hard_stop_after_successful_tool=True and a tool already succeeded, we block further tool calls.
    """

    if turn_id is None:
        turn_id = f"turn_{int(time.time() * 1000)}"

    # ------------------------
    # Helpers
    # ------------------------

    def _log(kind: str, content: str, meta: Dict[str, Any]) -> None:
        if add_event:
            add_event("system", content, {"channel": channel, "kind": kind, "turn_id": turn_id, **meta})

    def _log_assistant_final(text: str) -> None:
        if add_event and log_final_answer_event:
            add_event("assistant", text, {"channel": channel, "kind": "final_answer", "turn_id": turn_id})

    def _finalize_and_return(text: str) -> Tuple[str, List[Message]]:
        final_text = (text or "").strip() or "No Output"
        if canonical_answer is not None and canonical_answer not in final_text:
            final_text = canonical_answer

        if stream_callback:
            stream_callback(final_text)

        _log_assistant_final(final_text)
        return final_text, messages

    def _try_autorun_tool_from_user_input() -> Optional[Tuple[str, Dict[str, Any]]]:
        """Tiny heuristic: only for cases where user explicitly demands a tool."""
        if not tool_runtime or not tool_runtime.handlers:
            return None

        u = (user_input or "").strip()
        ul = u.lower()

        if "time" in ul and "time_now" in tool_runtime.handlers:
            return ("time_now", {})

        if "add" in tool_runtime.handlers:
            import re
            m = re.search(r"(-?\d+)\s*\+\s*(-?\d+)", ul)
            if m:
                return ("add", {"a": int(m.group(1)), "b": int(m.group(2))})

        return None

    def _execute_tool(
        tool_name: str,
        args: Dict[str, Any],
        tool_step: int,
    ) -> Tuple[bool, Any, Optional[str], int, str]:
        """Exec tool with validation + standardized tool_result text."""
        started = time.time()
        ok = True
        result: Any = None
        error: Optional[str] = None

        try:
            if not tool_runtime or tool_name not in tool_runtime.handlers:
                raise ValueError(f"unknown tool: {tool_name}")

            schema = tool_runtime.schemas.get(tool_name) if tool_runtime else None
            validate_args(schema, args)
            result = tool_runtime.handlers[tool_name](args)
        except Exception as e:
            ok = False
            error = str(e)

        latency_ms = int((time.time() - started) * 1000)
        tool_result_text = format_tool_result(
            tool_name,
            ok=ok,
            result=result,
            error=error,
            latency_ms=latency_ms,
        )

        _log(
            "tool_result",
            tool_result_text,
            {"tool": tool_name, "tool_step": tool_step, "ok": ok, "latency_ms": latency_ms},
        )

        return ok, result, error, latency_ms, tool_result_text

    # ------------------------
    # State
    # ------------------------

    tool_step = 0
    enforced_once = False
    blocked_spontaneous_once = False

    tool_executed = False          # a tool call happened (even if failed)
    tool_succeeded = False         # at least one tool returned ok=True
    tool_required = _user_demands_tool(user_input)

    # Gate for tool availability this turn
    tools_allowed = bool(tool_required or allow_spontaneous_tools)

    canonical_answer: Optional[str] = None

    # ------------------------
    # Autorun (ONLY if tool is required)
    # ------------------------
    if tool_required:
        autorun = _try_autorun_tool_from_user_input()
        if autorun is not None:
            tool_name, args = autorun

            _log(
                "tool_call",
                f"{TOOL_CALL_PREFIX} {tool_name} {json.dumps(args, ensure_ascii=False)}",
                {"tool": tool_name, "tool_step": 1},
            )

            ok, result, error, latency_ms, tool_result_text = _execute_tool(tool_name, args, tool_step=1)

            tool_executed = True
            tool_succeeded = tool_succeeded or ok

            extracted = extract_canonical_answer(result)
            if extracted is not None:
                canonical_answer = extracted

            # model must see tool_result
            messages = messages + [{"role": "system", "content": tool_result_text}]

            # reinforce canonical answer
            if canonical_answer is not None:
                messages = messages + [{
                    "role": "system",
                    "content": (
                        "A tool has produced an authoritative answer.\n"
                        "You MUST preserve the following value EXACTLY.\n\n"
                        f"CANONICAL_ANSWER: {canonical_answer}\n\n"
                        "Rules:\n"
                        "- You may add style, tone, or commentary.\n"
                        "- You MUST include the canonical answer verbatim.\n"
                        "- Do NOT change numbers, times, or wording.\n"
                    ),
                }]

            messages = messages + [{"role": "system", "content": FINALIZE_SYSTEM_PROMPT}]
            # Continue into model call for final answer; spiral guard below will stop further tools if configured.

    # ------------------------
    # Main loop
    # ------------------------
    while True:
        if tool_step >= max_tool_steps:
            # If we already have canonical, return it instead of dying noisily.
            if canonical_answer is not None:
                return _finalize_and_return(canonical_answer)
            raise RuntimeError("tool loop exceeded max_tool_steps (possible infinite loop)")

        # model response
        chunks: List[str] = []
        for token in engine.chat_stream(messages):
            chunks.append(token)
        assistant_text = ("".join(chunks)).strip() or "No Output"

        # parse tool call
        try:
            # Relaxed only if you explicitly allow it (agent/sim mode),
            # or if you want to be forgiving while bootstrapping.
            tc = parse_tool_call(
                assistant_text,
                relaxed=allow_spontaneous_tools  # or (tool_required and not enforced_once) if you want
            )
        except Exception as e:
            _log("tool_error", f"tool_call_parse_error: {str(e)}", {"tool_step": tool_step})
            tc = None

        # ------------------------
        # If model tried to tool-call but tools are NOT allowed this turn: block it
        # ------------------------
        if tc and not tools_allowed:
            # We give the model ONE chance to comply and answer normally.
            if not blocked_spontaneous_once:
                blocked_spontaneous_once = True
                messages = messages + [
                    # keep its text so you can debug why it wanted a tool
                    {"role": "assistant", "content": assistant_text},
                    {"role": "system", "content": (
                        "Tool calls are NOT allowed in this turn.\n"
                        "Reply normally without calling any tool.\n"
                        "Do NOT output /tool_call.\n"
                    )},
                ]
                continue

            # If it insists, just return a safe answer.
            return _finalize_and_return("ERROR: Tool call attempted but tools are disabled for this turn.")

        # ------------------------
        # No tool call detected
        # ------------------------
        if not tc:
            # Tool required but not executed -> enforce once
            if tool_required and not tool_executed:
                if not tool_runtime or not tool_runtime.handlers:
                    return _finalize_and_return(
                        "ERROR: Tool usage was requested, but no tools are available/registered for this turn."
                    )

                if not enforced_once:
                    enforced_once = True
                    messages = messages + [
                        {"role": "assistant", "content": assistant_text},
                        {"role": "system", "content": "A tool is required for this request. Output a /tool_call now."},
                    ]
                    continue

                return _finalize_and_return("ERROR: Tool required, but the model refused to call it.")

            # Normal completion (canonical enforced)
            return _finalize_and_return(assistant_text)

        # ------------------------
        # Tool call detected (and allowed)
        # ------------------------
        tool_name, args = tc

        # ✅ HARD SPIRAL STOP (only after a SUCCESSFUL tool)
        if hard_stop_after_successful_tool and tool_succeeded:
            if canonical_answer is not None:
                return _finalize_and_return(canonical_answer)
            return _finalize_and_return("Done. (A tool already succeeded; further tools are blocked.)")

        tool_step += 1

        _log(
            "tool_call",
            f"{TOOL_CALL_PREFIX} {tool_name} {json.dumps(args, ensure_ascii=False)}",
            {"tool": tool_name, "tool_step": tool_step},
        )

        # validator hook
        if tool_runtime and tool_runtime.validate_call:
            ok_v, err_v, patched = tool_runtime.validate_call(tool_name, args)
            if not ok_v:
                tool_result_text = format_tool_result(
                    tool_name, ok=False, result=None, error=err_v or "tool_call rejected", latency_ms=0
                )
                _log(
                    "tool_result",
                    tool_result_text,
                    {"tool": tool_name, "tool_step": tool_step, "ok": False, "latency_ms": 0},
                )

                tool_executed = True

                messages = messages + [
                    {"role": "assistant", "content": assistant_text},
                    {"role": "system", "content": tool_result_text},
                    {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
                ]
                continue

            if patched is not None:
                args = patched

        ok, result, error, latency_ms, tool_result_text = _execute_tool(tool_name, args, tool_step=tool_step)

        tool_executed = True
        tool_succeeded = tool_succeeded or ok

        # update canonical answer if extractable
        new_canonical = extract_canonical_answer(result)
        if new_canonical is not None:
            canonical_answer = new_canonical

        messages = messages + [
            {"role": "assistant", "content": assistant_text},
            {"role": "system", "content": tool_result_text},
        ]

        # reinforce canonical
        if canonical_answer is not None:
            messages = messages + [{
                "role": "system",
                "content": (
                    "A tool has produced an authoritative answer.\n"
                    "You MUST preserve the following value EXACTLY.\n\n"
                    f"CANONICAL_ANSWER: {canonical_answer}\n\n"
                    "Rules:\n"
                    "- You may add style, tone, or commentary.\n"
                    "- You MUST include the canonical answer verbatim.\n"
                    "- Do NOT change numbers, times, or wording.\n"
                ),
            }]

        messages = messages + [{"role": "system", "content": FINALIZE_SYSTEM_PROMPT}]
