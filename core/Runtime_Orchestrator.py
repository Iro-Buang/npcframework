from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

Message = Dict[str, str]
ToolHandler = Callable[[Dict[str, Any]], Any]

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


# -----------------------
# Parsing / formatting
# -----------------------

def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Strict enough to work, forgiving enough to survive LLMs.

    Expected start:
      /tool_call <tool_name> <json_payload>

    Accepts extra trailing text AFTER the JSON (we ignore it),
    but we always parse the first JSON object.
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

    # Parse first JSON object from 'rest' even if there's trailing junk like "= 7"
    rest = rest.strip()
    decoder = json.JSONDecoder()
    try:
        obj, idx = decoder.raw_decode(rest)
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
) -> Tuple[str, List[Message]]:
    """
    Executes inference. If model emits /tool_call..., executes tool, logs I/O,
    appends tool_result back into messages, then forces a finalize step so
    the final reply is clean (no /tool_call or JSON).
    """
    steps = 0

    while True:
        steps += 1
        if steps > (max_tool_steps + 2):
            raise RuntimeError("tool loop exceeded max_tool_steps (possible infinite loop)")

        # Collect full assistant output (tools + streaming don't mix cleanly)
        chunks: List[str] = []
        for token in engine.chat_stream(messages):
            chunks.append(token)
        assistant_text = "".join(chunks).strip() or "No Output"

        # If no tool call, return final reply
        tc = None
        try:
            tc = parse_tool_call(assistant_text)
        except Exception as e:
            if add_event:
                add_event("system", f"tool_call_parse_error: {str(e)}", {"channel": channel})
            tc = None

        if not tc:
            if stream_callback:
                stream_callback(assistant_text)
            return assistant_text, messages

        # Tool call detected
        tool_name, args = tc
        trace_id = f"{int(time.time()*1000)}_{steps}"

        if add_event:
            add_event(
                "system",
                f"{TOOL_CALL_PREFIX} {tool_name} {json.dumps(args, ensure_ascii=False)}",
                {"channel": channel, "kind": "tool_call", "tool": tool_name, "trace_id": trace_id},
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
                    "trace_id": trace_id,
                    "ok": ok,
                    "latency_ms": latency_ms,
                },
            )

        # Append the tool call + tool result to history
        messages = messages + [
            {"role": "assistant", "content": assistant_text},
            {"role": "system", "content": tool_result_text},
            {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
        ]

        # Now loop again; the next assistant response should be the clean final answer
