from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

Message = Dict[str, str]
ToolHandler = Callable[[Dict[str, Any]], Any]
ToolValidator = Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str], Optional[Dict[str, Any]]]]

TOOL_CALL_PREFIX = "/tool_call"
TOOL_RESULT_PREFIX = "/tool_result"

FINALIZE_SYSTEM_PROMPT = (
    "Tool execution is complete. Now answer the user's request.\n"
    "Rules:\n"
    "- Do NOT output /tool_call\n"
    "- Do NOT output JSON\n"
    "- Do NOT explain your reasoning\n"
    "- Output ONLY the final answer requested by the user\n"
)

@dataclass
class ToolRuntime:
    handlers: Dict[str, ToolHandler]
    schemas: Dict[str, Dict[str, Any]]
    validate_call: Optional[ToolValidator] = None


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
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

    # ✅ NEW: allow /tool_call anywhere (Gemma loves to preface with chatter)
    if not s.startswith(TOOL_CALL_PREFIX):
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        hit = next((ln for ln in lines if ln.startswith(TOOL_CALL_PREFIX)), None)
        if not hit:
            return None
        s = hit  # parse only the tool_call line

    payload = s[len(TOOL_CALL_PREFIX):].strip()
    if not payload:
        raise ValueError("tool_call missing payload")

    TOOL_ALIASES = {
        "get_time": "time_now",
        "time": "time_now",
        "now": "time_now",
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

    # If the user asked for time, and time_now exists
    if "time" in u and "time_now" in tool_runtime.handlers:
        return ("time_now", {})

    # If the user asked to add and 'add' tool exists, try parse "A + B"
    if "add" in u and "add" in tool_runtime.handlers:
        import re
        m = re.search(r"(-?\d+)\s*\+\s*(-?\d+)", u)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            return ("add", {"a": a, "b": b})

    return None


def _user_demands_tool(user_input: str) -> bool:
    u = (user_input or "").lower()
    triggers = [
        "use a tool", "use tool", "call a tool", "tool call", "use your tool", "use the tool",
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
) -> Tuple[str, List[Message]]:
    """Run inference with tool-call support + deterministic autorun + canonical answer enforcement."""
    if turn_id is None:
        turn_id = f"turn_{int(time.time() * 1000)}"

    def _try_autorun_tool_from_user_input() -> Optional[Tuple[str, Dict[str, Any]]]:
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
                a = int(m.group(1))
                b = int(m.group(2))
                return ("add", {"a": a, "b": b})

        return None

    tool_step = 0
    enforced_once = False
    tool_executed = False
    tool_required = _user_demands_tool(user_input)

    # ✅ NEW: carry canonical answer across the whole turn
    canonical_answer: Optional[str] = None

    # Helper: enforce canonical answer right before returning
    def _finalize_and_return(text: str) -> Tuple[str, List[Message]]:
        final_text = text
        if canonical_answer is not None and canonical_answer not in (final_text or ""):
            final_text = canonical_answer

        if stream_callback:
            stream_callback(final_text)
        if add_event:
            add_event("assistant", final_text, {"channel": channel, "kind": "final_answer", "turn_id": turn_id})
        return final_text, messages

    # ------------------------
    # Autorun for obvious cases when user explicitly demands a tool
    # ------------------------
    if tool_required:
        autorun = _try_autorun_tool_from_user_input()
        if autorun is not None:
            tool_name, args = autorun

            if add_event:
                add_event(
                    "system",
                    f"{TOOL_CALL_PREFIX} {tool_name} {json.dumps(args, ensure_ascii=False)}",
                    {"channel": channel, "kind": "tool_call", "tool": tool_name, "turn_id": turn_id, "tool_step": 1},
                )

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
            tool_result_text = format_tool_result(tool_name, ok=ok, result=result, error=error, latency_ms=latency_ms)

            if add_event:
                add_event(
                    "system",
                    tool_result_text,
                    {
                        "channel": channel,
                        "kind": "tool_result",
                        "tool": tool_name,
                        "turn_id": turn_id,
                        "tool_step": 1,
                        "ok": ok,
                        "latency_ms": latency_ms,
                    },
                )

            tool_executed = True

            # ✅ NEW: set canonical answer if extractable
            canonical_answer = extract_canonical_answer(result)

            # ✅ IMPORTANT: inject the ACTUAL tool_result into messages, so the model can see it
            messages = messages + [
                {"role": "system", "content": tool_result_text},
            ]

            # If we have a canonical answer, tell model it must include it verbatim
            if canonical_answer is not None:
                messages = messages + [
                    {
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
                    }
                ]

            messages = messages + [
                {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
            ]
            # Continue into loop; final answer will be canonical-enforced at return.

    # ------------------------
    # Main loop: model-driven tool calls
    # ------------------------
    while True:
        if tool_step > max_tool_steps:
            raise RuntimeError("tool loop exceeded max_tool_steps (possible infinite loop)")

        chunks: List[str] = []
        for token in engine.chat_stream(messages):
            chunks.append(token)
        assistant_text = "".join(chunks).strip() or "No Output"

        try:
            tc = parse_tool_call(assistant_text)
        except Exception as e:
            if add_event:
                add_event(
                    "system",
                    f"tool_call_parse_error: {str(e)}",
                    {"channel": channel, "kind": "tool_error", "turn_id": turn_id, "tool_step": tool_step},
                )
            tc = None

        if not tc:
            # Tool required but already executed => satisfied; just return (canonical enforced)
            if tool_required and tool_executed:
                return _finalize_and_return(assistant_text)

            # Tool required but none executed => enforce once then fail
            if tool_required:
                if not tool_runtime or not tool_runtime.handlers:
                    return _finalize_and_return("ERROR: Tool usage was requested, but no tools are available/registered for this turn.")

                if not enforced_once:
                    enforced_once = True
                    messages = messages + [
                        {"role": "assistant", "content": assistant_text},
                        {"role": "system", "content": "A tool is required for this request. Output a /tool_call now."},
                    ]
                    continue

                return _finalize_and_return("ERROR: Tool required, but the model refused to call it.")

            # Normal completion
            return _finalize_and_return(assistant_text)

        # Tool call detected
        tool_step += 1
        tool_name, args = tc

        if add_event:
            add_event(
                "system",
                f"{TOOL_CALL_PREFIX} {tool_name} {json.dumps(args, ensure_ascii=False)}",
                {"channel": channel, "kind": "tool_call", "tool": tool_name, "turn_id": turn_id, "tool_step": tool_step},
            )

        # Validator hook
        if tool_runtime and tool_runtime.validate_call:
            ok_v, err_v, patched = tool_runtime.validate_call(tool_name, args)
            if not ok_v:
                tool_result_text = format_tool_result(tool_name, ok=False, result=None, error=err_v or "tool_call rejected", latency_ms=0)

                if add_event:
                    add_event(
                        "system",
                        tool_result_text,
                        {
                            "channel": channel,
                            "kind": "tool_result",
                            "tool": tool_name,
                            "turn_id": turn_id,
                            "tool_step": tool_step,
                            "ok": False,
                            "latency_ms": 0,
                        },
                    )

                tool_executed = True
                # canonical stays as-is (no result)
                messages = messages + [
                    {"role": "assistant", "content": assistant_text},
                    {"role": "system", "content": tool_result_text},
                    {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
                ]
                continue

            if patched is not None:
                args = patched

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
        tool_result_text = format_tool_result(tool_name, ok=ok, result=result, error=error, latency_ms=latency_ms)

        if add_event:
            add_event(
                "system",
                tool_result_text,
                {
                    "channel": channel,
                    "kind": "tool_result",
                    "tool": tool_name,
                    "turn_id": turn_id,
                    "tool_step": tool_step,
                    "ok": ok,
                    "latency_ms": latency_ms,
                },
            )

        tool_executed = True

        # ✅ NEW: update canonical answer after ANY tool result
        # Only overwrite canonical_answer if we can extract one (keeps previous if None).
        new_canonical = extract_canonical_answer(result)
        if new_canonical is not None:
            canonical_answer = new_canonical

        messages = messages + [
            {"role": "assistant", "content": assistant_text},
            {"role": "system", "content": tool_result_text},
        ]

        # If we have canonical answer, reinforce it in prompt (helps model comply)
        if canonical_answer is not None:
            messages = messages + [
                {
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
                }
            ]

        messages = messages + [
            {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
        ]
