from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from npcframework.core.Runtime_Prompt_Compiler import ToolSpec


def time_now(_: dict) -> Dict[str, Any]:
    now = datetime.now().astimezone()
    return {
        "answer": now.strftime("%I:%M %p"),
        "meta": {
            "iso": now.isoformat(),
        },
    }


def add(args: dict) -> Dict[str, Any]:
    # Keep behavior flexible: accept int/float/str that can be casted
    a = float(args["a"])
    b = float(args["b"])
    s = a + b

    # Optional nicety: if it's effectively an int, store that too (doesn't affect canonical)
    as_int = int(s)
    is_integral = abs(s - as_int) < 1e-9

    return {
        "answer": as_int if is_integral else s,   # ✅ canonical
        "meta": {
            "a": a,
            "b": b,
            "is_integral": is_integral,
        },
    }


def builtin_toolset(
    allowlist: Optional[List[str]] = None,
) -> Tuple[List[ToolSpec], Dict[str, Any]]:
    all_tools: Dict[str, Tuple[ToolSpec, Any]] = {}

    # --- time_now ---
    all_tools["time_now"] = (
        ToolSpec(
            name="time_now",
            description="Get the current local system time.",
            schema={"type": "object", "properties": {}, "required": []},
        ),
        time_now,
    )

    # --- add ---
    all_tools["add"] = (
        ToolSpec(
            name="add",
            description="Add two numbers.",
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        ),
        add,  # ✅ real function, not lambda
    )

    if allowlist is None:
        selected = list(all_tools.keys())
    else:
        selected = [name for name in allowlist if name in all_tools]

    available_tools = [all_tools[name][0] for name in selected]
    tool_handlers = {name: all_tools[name][1] for name in selected}

    return available_tools, tool_handlers
