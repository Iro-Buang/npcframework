from pathlib import Path

def test_validate_kevin(validator):
    npc_dir = Path("npc/kevin.npc")
    report = validator(str(npc_dir))
    assert report.ok


def test_validate_anna(validator):
    npc_dir = Path("npc/anna.npc")
    report = validator(str(npc_dir))
    assert report.ok
