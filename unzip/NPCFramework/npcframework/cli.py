from __future__ import annotations

import argparse
from pathlib import Path

from npcframework.config import load_app_config

from npcframework.api import Engine, Session
from npcframework.core.NPC_Validator import validate_npc_dir
from npcframework.npcframework_types import EngineConfig, TurnInput


def cmd_debug(args: argparse.Namespace) -> int:
    """Compile and dump what would be sent to the model.

    We intentionally use the mock backend (no GPU required). The compiled messages
    are dumped through SessionConfig.debug_dump_*.
    """
    npc_dir = args.npc_dir
    session_cfg = None

    # tools (optional)
    available_tools = []
    tool_handlers = {}

    if args.config:
        c = load_app_config(args.config)
        npc_dir = c.npc_dir

        from npcframework.npcframework_types import SessionConfig

        s = dict(c.session)
        # Force dumps on for the debug command unless user already enabled them.
        session_cfg = SessionConfig(
            channel=str(s.get("channel", "cli")),
            environment_id=str(s.get("environment_id", "local")),
            environment_name=str(s.get("environment_name", "local")),
            history_limit=int(s.get("history_limit", 20)),
            include_state_in_prompt=bool(s.get("include_state_in_prompt", True)),
            include_tools_in_prompt=bool(s.get("include_tools_in_prompt", True)),
            include_perception_in_prompt=bool(s.get("include_perception_in_prompt", True)),
            include_memory_in_prompt=bool(s.get("include_memory_in_prompt", True)),
            debug_assert_messages_valid=bool(s.get("debug_assert_messages_valid", True)),
            debug_dump_dir=str(s.get("debug_dump_dir", ".npc/debug")),
            # Force dumps ON for debug command:
            debug_dump_messages_json=True,
            debug_dump_messages_txt=True,
        )

        tools_cfg = dict(getattr(c, "tools", {}) or {})
        if tools_cfg.get("enabled", False):
            allow = tools_cfg.get("allowlist", None)
            if allow is not None and not isinstance(allow, list):
                raise ValueError("tools.allowlist must be a list of tool names")
            from npcframework.tools.builtin import builtin_toolset
            available_tools, tool_handlers = builtin_toolset(allowlist=allow)

    else:
        # No config provided: use safe defaults and force dumps on.
        from npcframework.npcframework_types import SessionConfig
        session_cfg = SessionConfig(
            debug_dump_messages_json=True,
            debug_dump_messages_txt=True,
        )

    if not npc_dir:
        print("ERROR: missing npc_dir (provide positional npc_dir or --config)")
        return 2

    user_input = args.user_input or args.input
    if not user_input:
        print("ERROR: missing user input (provide positional user_input or --input)")
        return 2

    engine = Engine(EngineConfig(backend="mock"))
    session = Session(npc_dir=npc_dir, engine=engine, cfg=session_cfg)

    # Running a normal turn is fine; it will compile + dump messages.
    result = session.run_turn(TurnInput(user_input=user_input, available_tools=available_tools, tool_handlers=tool_handlers))
    if result.error:
        print("ERROR:", result.error)
        return 1

    print("Dumped compiled messages to:", session_cfg.debug_dump_dir)
    return 0


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
    # 1) load config (optional)
    npc_dir = args.npc_dir
    engine_cfg = EngineConfig(backend=args.backend)
    session_cfg = None

    # tools (optional)
    available_tools = []
    tool_handlers = {}

    if args.config:
        c = load_app_config(args.config)
        npc_dir = c.npc_dir

        # EngineConfig mapping (only fields provided will override defaults)
        e = dict(c.engine)
        engine_cfg = EngineConfig(
            backend=str(e.get("backend", args.backend)),
            model_path=str(e.get("model_path", "")),
            n_ctx=int(e.get("n_ctx", 8192)),
            n_threads=e.get("n_threads", None),
            n_gpu_layers=int(e.get("n_gpu_layers", 0)),
            temperature=float(e.get("temperature", 0.7)),
            top_p=float(e.get("top_p", 0.9)),
            max_tokens=int(e.get("max_tokens", 256)),
            stop=e.get("stop", None),
        )

        # SessionConfig mapping
        s = dict(c.session)
        from npcframework.npcframework_types import SessionConfig

        session_cfg = SessionConfig(
            channel=str(s.get("channel", "cli")),
            environment_id=str(s.get("environment_id", "local")),
            environment_name=str(s.get("environment_name", "local")),
            history_limit=int(s.get("history_limit", 20)),
            include_state_in_prompt=bool(s.get("include_state_in_prompt", True)),
            include_tools_in_prompt=bool(s.get("include_tools_in_prompt", True)),
            include_perception_in_prompt=bool(s.get("include_perception_in_prompt", True)),
            include_memory_in_prompt=bool(s.get("include_memory_in_prompt", True)),
            debug_assert_messages_valid=bool(s.get("debug_assert_messages_valid", True)),
            debug_dump_dir=str(s.get("debug_dump_dir", ".npc/debug")),
            debug_dump_messages_json=bool(s.get("debug_dump_messages_json", False)),
            debug_dump_messages_txt=bool(s.get("debug_dump_messages_txt", False)),
            default_state_mode=str(s.get("default_state_mode", "idle")),
            default_state_mood=str(s.get("default_state_mood", "neutral")),
            default_state_energy=float(s.get("default_state_energy", 0.8)),
            runtime_goal=str(s.get("runtime_goal", "help the user")),
            runtime_mode=str(s.get("runtime_mode", "conversational")),
            runtime_env_facts=list(s.get("runtime_env_facts", [])),
            runtime_env_rules=list(s.get("runtime_env_rules", [])),
            runtime_perception_facts=list(s.get("runtime_perception_facts", [])),
            runtime_additional_policies=list(s.get("runtime_additional_policies", [])),
            identity_role_append=str(s.get("identity_role_append", "")),
        )

        tools_cfg = dict(getattr(c, "tools", {}) or {})
        if tools_cfg.get("enabled", False):
            allow = tools_cfg.get("allowlist", None)
            if allow is not None and not isinstance(allow, list):
                raise ValueError("tools.allowlist must be a list of tool names")
            from npcframework.tools.builtin import builtin_toolset
            available_tools, tool_handlers = builtin_toolset(allowlist=allow)

    # 2) run
    if not npc_dir:
        print("ERROR: missing npc_dir (provide positional npc_dir or --config)")
        return 2

    user_input = args.user_input or args.input
    if not user_input:
        print("ERROR: missing user input (provide positional user_input or --input)")
        return 2

    engine = Engine(engine_cfg)
    session = Session(npc_dir=npc_dir, engine=engine, cfg=session_cfg)

    turn = TurnInput(user_input=user_input, available_tools=available_tools, tool_handlers=tool_handlers)
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
    pr.add_argument("npc_dir", nargs="?", help="Path to .npc directory (optional if --config is provided)")
    pr.add_argument("user_input", nargs="?", help="User input (optional if --input is provided)")
    pr.add_argument("--input", dest="input", default=None, help="User input")
    pr.add_argument("--backend", default="mock", choices=["mock", "llamacpp"], help="Inference backend (overridden by config)")
    pr.add_argument("--config", default=None, help="Config file (.toml/.json/.yaml)")
    pr.set_defaults(func=cmd_run)

    pd = sub.add_parser("debug", help="Compile and dump the messages that would be sent to the model")
    pd.add_argument("npc_dir", nargs="?", help="Path to .npc directory (optional if --config is provided)")
    pd.add_argument("user_input", nargs="?", help="User input (optional if --input is provided)")
    pd.add_argument("--input", dest="input", default=None, help="User input")
    pd.add_argument("--config", default=None, help="Config file (.toml/.json/.yaml)")
    pd.set_defaults(func=cmd_debug)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
