from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .npcframework_types import Message


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _messages_to_txt(messages: List[Message]) -> str:
    parts: List[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        parts.append(f"### {role.upper()}\n{content}\n")
    return "\n".join(parts).strip() + "\n"


def dump_messages(
    *,
    messages: List[Message],
    out_dir: str,
    turn_id: str,
    write_json: bool,
    write_txt: bool,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    """Write debug artifacts and return paths.

    This is intentionally simple: we dump what the engine sees (messages).
    """
    out = Path(out_dir)
    _ensure_dir(out)

    written: Dict[str, str] = {}

    if write_json:
        payload = {
            "meta": meta or {},
            "messages": messages,
        }
        p = out / f"{turn_id}.messages.json"
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written["messages_json"] = str(p)

    if write_txt:
        p = out / f"{turn_id}.messages.txt"
        p.write_text(_messages_to_txt(messages), encoding="utf-8")
        written["messages_txt"] = str(p)

    return written
