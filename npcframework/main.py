# from __future__ import annotations
#
# import argparse
# import sys
# from pathlib import Path
#
# from npcframework.core.NPC_Validator import validate_npc_dir
# from npcframework.core.Runtime_Inference_LlamaCPP import build_engine, RuntimeConfig, TurnRequest, run_turn
#
#
# def _cmd_validate(args: argparse.Namespace) -> int:
#     npc_dir = Path(args.npc_dir)
#     ok, err = validate_npc_dir(str(npc_dir))
#     if ok:
#         print(f"OK: {npc_dir}")
#         return 0
#     print(f"INVALID: {npc_dir}\n{err}")
#     return 2
#
#
# def _cmd_run(args: argparse.Namespace) -> int:
#     cfg = RuntimeConfig(
#         channel=args.channel,
#         environment_id=args.environment_id,
#         environment_name=args.environment_name,
#         debug_print_messages=args.debug,
#     )
#
#     engine = build_engine(cfg)
#
#     result = run_turn(
#         req=TurnRequest(
#             npc_dir=args.npc_dir,
#             user_input=args.user_input,
#         ),
#         cfg=cfg,
#         engine=engine,
#     )
#
#     if result.error:
#         print("ERROR:", result.error)
#         return 1
#
#     print(f"{result.npc_name}> {result.assistant_reply}")
#     return 0
#
#
# def build_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(prog="npcframework", description="NPCFramework CLI")
#     sub = p.add_subparsers(dest="cmd", required=True)
#
#     pv = sub.add_parser("validate", help="Validate an .npc directory")
#     pv.add_argument("npc_dir")
#     pv.set_defaults(fn=_cmd_validate)
#
#     pr = sub.add_parser("run", help="Run a single turn")
#     pr.add_argument("npc_dir")
#     pr.add_argument("user_input")
#     pr.add_argument("--channel", default="cli")
#     pr.add_argument("--environment-id", default="cli_env")
#     pr.add_argument("--environment-name", default="cli")
#     pr.add_argument("--debug", action="store_true")
#     pr.set_defaults(fn=_cmd_run)
#
#     return p
#
#
# def main(argv: list[str] | None = None) -> int:
#     parser = build_parser()
#     args = parser.parse_args(argv)
#     return int(args.fn(args))
#
#
# if __name__ == "__main__":
#     raise SystemExit(main())



from npcframework.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
