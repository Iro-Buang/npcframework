from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from db import Event

Message = Dict[str, str]


@dataclass
class CompileOptions:
    history_limit: int = 20          # number of events pulled from db
    include_state: bool = True
    state_keys: Optional[List[str]] = None  # if None, include a small default set


def _bullet(lines: List[str]) -> str:
    return "\n".join(f"- {x}" for x in lines)


def build_system_prompt(identity: Dict[str, Any], persona: Dict[str, Any], policy: Dict[str, Any]) -> str:
    # identity
    ident_lines = []
    if "archetype" in identity:
        ident_lines.append(f"Archetype: {identity['archetype']}")
    if "description" in identity:
        ident_lines.append(f"Description: {identity['description']}")

    core_values = identity.get("core_values") or []
    purpose = identity.get("purpose") or []

    # persona
    tone = persona.get("tone", "neutral")
    style = persona.get("style", "neutral")
    verbosity = persona.get("verbosity", "medium")
    humor = persona.get("humor", "none")

    speech_rules = persona.get("speech_rules") or []
    taboos = persona.get("taboos") or []

    # policy
    boundaries = policy.get("boundaries") or []
    refusal_policy = policy.get("refusal_policy") or []
    truthfulness = policy.get("truthfulness") or []

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

    parts.append("When replying: stay in character, be useful, and avoid fluff.\n")

    return "\n".join(parts).strip()


def build_state_prompt(state_snapshot: Dict[str, Any]) -> str:
    # Keep this short. LLMs hate walls of state.
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
        # pass through only supported roles for now
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
    current_user_input: str,
    state_snapshot: Optional[Dict[str, Any]] = None,
    options: Optional[CompileOptions] = None,
) -> List[Message]:
    if options is None:
        options = CompileOptions()

    system_prompt = build_system_prompt(identity, persona, policy)
    messages: List[Message] = [{"role": "system", "content": system_prompt}]

    if options.include_state and state_snapshot:
        messages.append({"role": "system", "content": build_state_prompt(state_snapshot)})

    # History
    history_msgs = events_to_messages(recent_events[-options.history_limit:])
    messages.extend(history_msgs)

    # Current user input (always last)
    messages.append({"role": "user", "content": current_user_input})

    return messages
