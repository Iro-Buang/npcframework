import shutil
from pathlib import Path
from typing import Iterable, List, Dict

from npcframework.api import Session, Engine
from npcframework.npcframework_types import TurnInput, ToolSpec, EngineConfig

Message = Dict[str, str]


class ToolCallMockEngine:
    def chat_stream(self, messages: List[Message]) -> Iterable[str]:
        # Emit the exact contract your tool parser expects
        yield '/tool_call {"tool":"add","args":{"a":2,"b":3}}'

    def chat(self, messages: List[Message]) -> str:
        return "".join(self.chat_stream(messages))


def test_tool_call_executes(tmp_path):
    src = Path("npc/kevin.npc")
    dst = tmp_path / "kevin.npc"
    shutil.copytree(src, dst)

    # Provide tool + handler
    add_tool = ToolSpec(
        name="add",
        description="Add two numbers",
        schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        few_shots=[],
    )

    def add_handler(args):
        return args["a"] + args["b"]

    engine = Engine(EngineConfig(backend="mock"))
    engine.backend = ToolCallMockEngine()
    session = Session(npc_dir=str(dst), engine=engine)

    result = session.run_turn(
        TurnInput(
            user_input="Use tool add with a=2 and b=3, then tell me the result.",
            available_tools=[add_tool],
            tool_handlers={"add": add_handler},
        )
    )

    assert result.error is None
    assert result.assistant_reply is not None
    # Optional if you store traces
    # assert any(tc.tool == "add" for tc in result.trace.tool_calls)
