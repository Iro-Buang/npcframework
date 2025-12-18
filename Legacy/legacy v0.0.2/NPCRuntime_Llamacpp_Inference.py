from __future__ import annotations

import time
from typing import Dict, Any

from NPCLoader import load_npc
from NPC_DB_Manager import NPCDatabase
from NPCPrompt_Compiler import compile_messages, CompileOptions, RuntimeInjection, ToolSpec
from NPCCommands import handle_command
from NPC_DB_Episodic_Promoter import EpisodicPromoter

from inference.llamacpp import LlamaCppEngine, LlamaCppConfig


def _debug_print_messages(messages):
    print("\n===== RAW MESSAGES SENT TO MODEL =====\n")
    for i, m in enumerate(messages, 1):
        role = m.get("role", "?").upper()
        content = m.get("content", "")
        print(f"[{i}] {role}\n{content}\n{'-'*60}")
    print("===== END RAW MESSAGES =====\n")


def debug_inject_prompt_override(messages, injected_prompt: str):
    """
    Ensures llama.cpp receives a proper chat message list,
    with the injected prompt as the system message.

    - If a system message exists, replace it.
    - If none exists, insert at the top.
    """
    if not injected_prompt or not injected_prompt.strip():
        return messages

    injected_prompt = injected_prompt.strip()

    # messages should be List[Dict[str, str]]
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list of dicts, got {type(messages)}")

    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages[0]["content"] = injected_prompt
        return messages

    # Find any system message later in the list and replace it
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            m["content"] = injected_prompt
            return messages

    # No system message found, insert one
    messages.insert(0, {"role": "system", "content": injected_prompt})
    return messages


def _ensure_default_state(db: NPCDatabase) -> None:
    if db.get_state("mode") is None:
        db.set_state("mode", "idle")
    if db.get_state("mood") is None:
        db.set_state("mood", "neutral")
    if db.get_state("energy") is None:
        db.set_state("energy", 0.8)


def _state_snapshot(db: NPCDatabase) -> Dict[str, Any]:
    return {
        "mode": db.get_state("mode", "idle"),
        "mood": db.get_state("mood", "neutral"),
        "energy": db.get_state("energy", 0.8),
    }


def _sanitize_user_input(text: str, npc_name: str) -> str:
    """
    UX guard: if user types '<NPC_NAME>> blah', treat it as just 'blah'.
    """
    t = text.strip()
    prefix = f"{npc_name.lower()}>"
    if t.lower().startswith(prefix):
        t = t.split(">", 1)[1].strip()
    return t

def run_cli(npc_dir: str, *, history_limit: int = 20) -> None:
    channel = "cli"
    environment_id = "local_cli"  # or "mad_dog", "discord", "web", etc.


    npc = load_npc(npc_dir)

    db = NPCDatabase(npc.paths.db)
    db.init_db()
    _ensure_default_state(db)

    engine = LlamaCppEngine(
        LlamaCppConfig(
            model_path="inference/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
            n_ctx=8192,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            n_gpu_layers=0,  # CPU for now
        )
    )

    npc_id = npc.manifest.get("id")
    npc_name = npc.manifest.get("display_name")
    print(f"{npc_name} online (llama.cpp). NPC={npc_id}  DB={npc.paths.db}")
    print("Type /help for commands.\n")

    while True:
        raw = input("You> ")
        if raw is None:
            continue

        user_input = _sanitize_user_input(raw, npc_name)
        if not user_input:
            continue

        # System1: commands (no model)
        cmdres = handle_command(user_input, npc=npc, npc_db=db)
        if cmdres.handled:
            if cmdres.response:
                print(f"{npc_name}> {cmdres.response}\n")
            if cmdres.should_exit:
                break
            continue

        # Log user message first (DB is truth)
        db.add_event(
            "user",
            user_input,
            meta={
                "channel": channel,
                "environment_id": environment_id,
                "event_type": "message",
                "modality": "text",
                "content_format": "plain",
                # optional: you can add "importance_hint": 10
            },
        )

        # Compile messages for model
        recent = db.get_recent_events(limit=history_limit)
        state = _state_snapshot(db)

        runtime = RuntimeInjection(
            environment_name="local_cli",
            environment_facts=[
                "Conversation is text-only.",
                "Only the most recent messages are available in the prompt.",
            ],
            environment_rules=[
                "Environment facts must be objective; no emotion or intent.",
            ],
            state={
                "mode": "conversational",
                "goal": "help the user",
                "energy": 0.8,
            },
            perception_facts=[
                "User greeted Kevin.",
            ],
            # tool promotion example:
            promote_tools=("search" in user_input.lower()),
            available_tools=[ToolSpec(
                name="search",
                description="Search the web for factual info.",
                schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                few_shots=[{"input": "Search Bacolod condos",
                            "output": "/tool_call search {\"query\":\"Bacolod condo price\"}"}],
            )],
            additional_policies=[
                "Do not claim you executed a tool unless the runtime confirms it.",
            ],
            identity_role_append="Kevin is currently operating in a local CLI sandbox.",
        )
        messages = compile_messages(
            identity=npc.identity,
            persona=npc.persona,
            policy=npc.policy,
            recent_events=recent,
            runtime=runtime,
            options=CompileOptions(history_limit=20, include_state=True, include_tools=True),
        )

        # DEBUG: inspect exactly what the model will see
        _debug_print_messages(messages)

        # Stream response
        print(f"{npc_name}> ", end="", flush=True)


        chunks = []
        t0 = time.time()
        showed_wait = False

        try:
            for piece in engine.chat_stream(messages):
                # If first token is slow, show a subtle "..."
                if not showed_wait and (time.time() - t0) > 0.9 and not chunks:
                    print("...", end="", flush=True)
                    showed_wait = True

                print(piece, end="", flush=True)
                chunks.append(piece)


        except KeyboardInterrupt:
            print(f"\n{npc_name}> (interrupted)\n")
            continue

        except Exception as e:
            print(f"\n{npc_name}> (inference error: {e})\n")
            continue

        reply = "".join(chunks).strip()
        print("\n")  # end the line after streaming

        # Log npc reply
        if reply:
            db.add_event(
                "assistant",
                reply,
                meta={
                    "channel": channel,
                    "environment_id": environment_id,
                    "event_type": "message",
                    "modality": "text",
                    "content_format": "plain",
                    # optional: "status": "processed"
                },
            )

        else:
            db.add_event(
                "assistant",
                "No Output",
                meta={
                    "channel": channel,
                    "environment_id": environment_id,
                    "event_type": "message",
                    "modality": "text",
                    "content_format": "plain",
                    # optional: "status": "processed"
                },
            )

        promoter = EpisodicPromoter(db)
        promoter.promote()



if __name__ == "__main__":
    import sys
    npc_dir = sys.argv[1] if len(sys.argv) > 1 else "npc/kevin.npc"
    run_cli(npc_dir, history_limit=20)
