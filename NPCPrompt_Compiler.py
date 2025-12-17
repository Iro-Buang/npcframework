from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable

from NPC_DB_Manager import Event

Message = Dict[str, str]


@dataclass
class CompileOptions:
    history_limit: int = 20
    include_state: bool = True


def _bullet(lines: List[str]) -> str:
    return "\n".join(f"- {x}" for x in lines)


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def _get_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    meta is allowed everywhere. If missing or wrong type, treat as empty.
    """
    m = doc.get("meta")
    return m if isinstance(m, dict) else {}


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge dictionaries:
    - dict + dict => merge recursively
    - otherwise => incoming overrides base
    Returns a new dict.
    """
    out: Dict[str, Any] = dict(base)
    for k, v in incoming.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _merge_meta_in_order(docs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic precedence: later docs override earlier ones.
    """
    merged: Dict[str, Any] = {}
    for doc in docs:
        merged = _deep_merge(merged, _get_meta(doc))
    return merged


def _render_meta(meta: Dict[str, Any]) -> str:
    """
    Render meta in a structured, readable way.
    Prefer YAML if available; fall back to an indented key/value dump.
    """
    if not meta:
        return ""

    # If user chooses to separate meta into prompt/runtime, prefer meta.prompt.
    # But do NOT require it for v0.1 — if absent, include everything.
    payload = meta.get("prompt")
    if isinstance(payload, dict):
        meta_for_prompt = payload
    else:
        meta_for_prompt = meta

    if not meta_for_prompt:
        return ""

    # Try YAML dump if PyYAML exists (your project already uses it in validator).
    try:
        import yaml  # type: ignore
        dumped = yaml.safe_dump(meta_for_prompt, sort_keys=False, allow_unicode=True).strip()
        return dumped
    except Exception:
        # Fallback: simple nested dump
        lines: List[str] = []

        def walk(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{prefix}{k}:")
                        walk(v, prefix=prefix + "  ")
                    else:
                        lines.append(f"{prefix}{k}: {_as_str(v)}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        lines.append(f"{prefix}-")
                        walk(item, prefix=prefix + "  ")
                    else:
                        lines.append(f"{prefix}- {_as_str(item)}")
            else:
                lines.append(f"{prefix}{_as_str(obj)}")

        walk(meta_for_prompt)
        return "\n".join(lines).strip()


def build_system_prompt(
    identity: Dict[str, Any],
    persona: Dict[str, Any],
    policy: Dict[str, Any],
    *,
    # v0.1: accept optional extra docs, so you can later pass memory/kernel/actions/tools
    extra_docs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    ident_lines: List[str] = []
    if "archetype" in identity:
        ident_lines.append(f"Archetype: {identity['archetype']}")
    if "description" in identity:
        ident_lines.append(f"Description: {identity['description']}")

    core_values = identity.get("core_values") or []
    purpose = identity.get("purpose") or []

    tone = persona.get("tone", "neutral")
    style = persona.get("style", "neutral")
    verbosity = persona.get("verbosity", "medium")
    humor = persona.get("humor", "none")

    speech_rules = persona.get("speech_rules") or []
    taboos = persona.get("taboos") or []

    boundaries = policy.get("boundaries") or []
    refusal_policy = policy.get("refusal_policy") or []
    truthfulness = policy.get("truthfulness") or []

    # ---- META MERGE (v0.1 extension bus) ----
    docs_in_precedence: List[Dict[str, Any]] = [identity, persona, policy]
    if extra_docs:
        docs_in_precedence.extend(extra_docs)

    merged_meta = _merge_meta_in_order(docs_in_precedence)
    meta_text = _render_meta(merged_meta)

    parts: List[str] = []

    parts.append("You are an NPC with a stable identity. Follow the identity/persona/policy below.\n")

    if ident_lines:
        parts.append("IDENTITY\n" + "\n".join(ident_lines) + "\n")

    if core_values:
        parts.append("CORE VALUES\n" + _bullet(core_values) + "\n")

    if purpose:
        parts.append("PURPOSE\n" + _bullet(purpose) + "\n")

    parts.append("PERSONA\n" + "\n".join([
        f"Tone: {tone}",
        f"Style: {style}",
        f"Verbosity: {verbosity}",
        f"Humor: {humor}",
    ]) + "\n")

    if speech_rules:
        parts.append("SPEECH RULES\n" + _bullet(speech_rules) + "\n")

    if taboos:
        parts.append("TABOOS\n" + _bullet(taboos) + "\n")

    if boundaries:
        parts.append("BOUNDARIES\n" + _bullet(boundaries) + "\n")

    if truthfulness:
        parts.append("TRUTHFULNESS\n" + _bullet(truthfulness) + "\n")

    if refusal_policy:
        parts.append("REFUSAL POLICY\n" + _bullet(refusal_policy) + "\n")

    # Put meta close to the end so it has higher “recency” in the system prompt.
    if meta_text:
        parts.append("META (extensions)\n" + meta_text + "\n")
        parts.append("META ENFORCEMENT\n- META rules are binding when present.\n")

    parts.append("When replying: stay in character.\n")

    return "\n".join(parts).strip()


def build_state_prompt(state_snapshot: Dict[str, Any]) -> str:
    mode = state_snapshot.get("mode")
    mood = state_snapshot.get("mood")
    energy = state_snapshot.get("energy")

    lines = ["CURRENT STATE (runtime snapshot)"]
    if mode is not None:
        lines.append(f"- mode: {mode}")
    if mood is not None:
        lines.append(f"- mood: {mood}")
    if energy is not None:
        lines.append(f"- energy: {energy}")

    return "\n".join(lines)


def events_to_messages(events: List[Event]) -> List[Message]:
    msgs: List[Message] = []
    for e in events:
        role = e.role
        if role not in ("user", "assistant", "system"):
            role = "user"
        msgs.append({"role": role, "content": e.content})
    return msgs


def compile_messages(
    *,
    identity: Dict[str, Any],
    persona: Dict[str, Any],
    policy: Dict[str, Any],
    recent_events: List[Event],
    state_snapshot: Optional[Dict[str, Any]] = None,
    options: Optional[CompileOptions] = None,
    # future proof: let caller pass memory/kernel/actions/tools docs for meta aggregation
    extra_docs: Optional[List[Dict[str, Any]]] = None,
) -> List[Message]:
    """
    Compile model messages from:
    - system prompt (identity/persona/policy + merged meta from all docs)
    - optional runtime state prompt
    - recent events from DB (single source of truth)

    IMPORTANT: current user input should already be logged in DB before calling this.
    """
    if options is None:
        options = CompileOptions()

    system_prompt = build_system_prompt(identity, persona, policy, extra_docs=extra_docs)
    messages: List[Message] = [{"role": "system", "content": system_prompt}]

    # if options.include_state and state_snapshot:
    #     messages.append({"role": "system", "content": build_state_prompt(state_snapshot)})

    history_msgs = events_to_messages(recent_events[-options.history_limit:])
    messages.extend(history_msgs)

    return messages
