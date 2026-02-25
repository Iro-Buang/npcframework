import pytest
from npcframework.core.NPC_Validator import validate_npc_dir

@pytest.fixture
def validator():
    return validate_npc_dir
