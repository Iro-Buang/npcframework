from __future__ import annotations

import argparse
from pathlib import Path

from npcframework.api import Engine, Session
from npcframework.core.NPC_Validator import validate_npc_dir
from npcframework.npcframework_types import EngineConfig, TurnInput


def cmd_validate(args: argparse.Namespace) -> int:
    npc_dir = Path(args.npc_dir)
    report = validate_npc_dir(str(npc_dir))

    if report.ok:
        print(f"OK: {report.summary()}")
        return 0

    print(f"INVALID: {report.summary()}")
    for issue in report.issues:
        loc = f" [{issue.path}]" if issue.path else ""
        print(f"- {issue.severity} {issue.code}{loc}: {issue.message}")
    return 2


def cmd_run(args: argparse.Namespace) -> int:
    engine = Engine(EngineConfig(backend=args.backend))
    session = Session(npc_dir=args.npc_dir, engine=engine)

    turn = TurnInput(user_input=args.user_input)
    result = session.run_turn(turn)
    if result.error:
        print("ERROR:", result.error)
        return 1

    print(f"{result.npc_name}> {result.assistant_reply}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="npcframework", description="NPCFramework CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="Validate an NPC directory")
    pv.add_argument("npc_dir")
    pv.set_defaults(func=cmd_validate)

    pr = sub.add_parser("run", help="Run a single NPC turn")
    pr.add_argument("npc_dir")
    pr.add_argument("user_input")
    pr.add_argument("--backend", default="mock", choices=["mock", "llamacpp"])
    pr.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
