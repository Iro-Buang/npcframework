from __future__ import annotations

import sys
from pathlib import Path

# Put /core on sys.path BEFORE importing anything from it
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))

from core.Runtime_Inference_LlamaCPP import run_turn, RuntimeConfig, TurnRequest, build_engine
# If you want CLI too:
# from Runtime_Inference_LlamaCPP import run_cli


def main_dump():
    cfg = RuntimeConfig(
        channel="test",
        environment_id="test_env",
        environment_name="unit_test",
        debug_print_messages=False,
    )

    engine = build_engine(cfg)

    result = run_turn(
        req=TurnRequest(
            npc_dir="npc/anna.npc",
            user_input="Hello Mike, who are you?",
            # no stream_callback => dump mode
        ),
        cfg=cfg,
        engine=engine,
    )

    if result.error:
        print("ERROR:", result.error)
        return

    print(f"{result.npc_name}> {result.assistant_reply}")


def main_stream():
    cfg = RuntimeConfig(
        channel="test",
        environment_id="test_env",
        environment_name="unit_test",
        debug_print_messages=False,
    )

    engine = build_engine(cfg)

    def on_token(tok: str) -> None:
        print(tok, end="", flush=True)

    result = run_turn(
        req=TurnRequest(
            npc_dir="npc/kevin.npc",
            user_input="Tell me a one-liner about NPCFramework.",
            stream_callback=on_token,  # streaming mode
        ),
        cfg=cfg,
        engine=engine,
    )

    print()  # newline after stream

    if result.error:
        print("ERROR:", result.error)
        return

    # Optional: print final (already streamed) or just trust the stream
    # print(f"{result.npc_name}> {result.assistant_reply}")


if __name__ == "__main__":
    # pick one:
    # main_dump()
    main_stream()
