import shutil
from pathlib import Path

from npcframework.api import Engine, Session
from npcframework.npcframework_types import EngineConfig, TurnInput


def test_one_turn_mock(tmp_path):
    # Copy NPC to tmp so we don't mutate real DBs
    src = Path("npc/kevin.npc")
    dst = tmp_path / "kevin.npc"
    shutil.copytree(src, dst)

    engine = Engine(EngineConfig(backend="mock"))
    session = Session(npc_dir=str(dst), engine=engine)

    result = session.run_turn(TurnInput(user_input="Hello"))

    assert result.error is None
    assert result.assistant_reply is not None
    assert isinstance(result.assistant_reply, str)
    assert result.assistant_reply.strip() != ""
