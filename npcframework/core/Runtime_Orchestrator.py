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
    # Optional trust boundary: app can approve/deny/patch tool calls.
    # Returns: (ok, error_msg, patched_args)
    validate_call: Optional[ToolValidator] = None


# -----------------------
# Parsing / formatting
# -----------------------

def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Expected start:
      /tool_call <tool_name> <json_payload>

    Parses the FIRST JSON object from payload, ignoring trailing junk.
    """
    if not text:
        return None

    s = text.strip()
    if not s.startswith(TOOL_CALL_PREFIX):
        return None

    parts = s.split(maxsplit=2)
    if len(parts) < 3:
        raise ValueError("tool_call missing tool_name or json_payload")

    _, tool_name, rest = parts
    tool_name = tool_name.strip()
    if not tool_name:
        raise ValueError("tool_call tool_name is empty")

    rest = rest.strip()
    decoder = json.JSONDecoder()
    try:
        obj, _idx = decoder.raw_decode(rest)
    except json.JSONDecodeError as e:
        raise ValueError(f"tool_call json invalid: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("tool_call json_payload must be a JSON object (dict)")

    return tool_name, obj


def format_tool_result(
    tool_name: str,
    *,
    ok: bool,
    result: Any = None,
    error: Optional[str] = None,
    latency_ms: Optional[int] = None,
) -> str:
    payload: Dict[str, Any] = {
        "ok": ok,
        "tool": tool_name,
        "result": result,
        "error": error,
        "latency_ms": latency_ms,
    }
    return f"{TOOL_RESULT_PREFIX} {tool_name} {json.dumps(payload, ensure_ascii=False)}"


# -----------------------
# Minimal schema checks
# -----------------------

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
            raise ValueError(f"field '{k}' must be {t}")


# -----------------------
# Orchestration loop
# -----------------------

def run_with_tools(
    *,
    engine: Any,
    messages: List[Message],
    tool_runtime: Optional[ToolRuntime],
    max_tool_steps: int = 5,
    stream_callback: Optional[Callable[[str], None]] = None,
    add_event: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    channel: str = "runtime",
    turn_id: Optional[str] = None,   # 🔥 NEW (optional, app-controlled)
) -> Tuple[str, List[Message]]:
    """
    Executes inference with tool support.

    Enhancements (drop-in, backward compatible):
    - turn_id: per-turn traceability
    - tool_step: ordered tool execution within a turn
    """

    if turn_id is None:
        # Deterministic-enough fallback, but app SHOULD provide this
        turn_id = f"turn_{int(time.time() * 1000)}"

    tool_step = 0
    steps = 0

    while True:
        steps += 1
        if steps > (max_tool_steps + 2):
            raise RuntimeError("tool loop exceeded max_tool_steps (possible infinite loop)")

        # ---- Run model (collect full output) ----
        chunks: List[str] = []
        for token in engine.chat_stream(messages):
            chunks.append(token)
        assistant_text = "".join(chunks).strip() or "No Output"

        # ---- Attempt to parse tool call ----
        try:
            tc = parse_tool_call(assistant_text)
        except Exception as e:
            if add_event:
                add_event(
                    "system",
                    f"tool_call_parse_error: {str(e)}",
                    {
                        "channel": channel,
                        "kind": "tool_error",
                        "turn_id": turn_id,
                        "tool_step": tool_step,
                    },
                )
            tc = None

        # ---- No tool call → final answer ----
        if not tc:
            if stream_callback:
                stream_callback(assistant_text)

            if add_event:
                add_event(
                    "assistant",
                    assistant_text,
                    {
                        "channel": channel,
                        "kind": "final_answer",
                        "turn_id": turn_id,
                    },
                )

            return assistant_text, messages

        # ---- Tool call detected ----
        tool_step += 1
        tool_name, args = tc

        if add_event:
            add_event(
                "system",
                f"{TOOL_CALL_PREFIX} {tool_name} {json.dumps(args, ensure_ascii=False)}",
                {
                    "channel": channel,
                    "kind": "tool_call",
                    "tool": tool_name,
                    "turn_id": turn_id,
                    "tool_step": tool_step,
                },
            )

        # ---- Trust boundary (APP controls this) ----
        if tool_runtime and tool_runtime.validate_call:
            ok_v, err_v, patched = tool_runtime.validate_call(tool_name, args)
            if not ok_v:
                tool_result_text = format_tool_result(
                    tool_name,
                    ok=False,
                    result=None,
                    error=err_v or "tool_call rejected",
                    latency_ms=0,
                )

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

                messages = messages + [
                    {"role": "assistant", "content": assistant_text},
                    {"role": "system", "content": tool_result_text},
                    {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
                ]
                continue

            if patched is not None:
                args = patched

        # ---- Execute tool ----
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

        # ---- Feed result back into context ----
        messages = messages + [
            {"role": "assistant", "content": assistant_text},
            {"role": "system", "content": tool_result_text},
            {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
        ]

