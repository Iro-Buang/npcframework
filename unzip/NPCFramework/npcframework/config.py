from __future__ import annotations

"""NPCFramework config loader.

Goal: eliminate hardcoded presets in scripts/CLI.

We intentionally keep this dependency-light:
- TOML: built-in via tomllib (Python 3.11+)
- JSON: built-in
- YAML: optional (requires PyYAML); if unavailable, YAML configs raise a clear error.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json


@dataclass
class AppConfig:
    """Top-level config used by the CLI and simple apps."""

    npc_dir: str
    engine: Dict[str, Any]
    session: Dict[str, Any]
    tools: Dict[str, Any] = field(default_factory=dict)


def _load_toml(path: Path) -> Dict[str, Any]:
    import tomllib  # py311+

    return tomllib.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "YAML config requires PyYAML. Install with: pip install pyyaml"
        ) from e

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object")
    return data


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object")
    return data


def load_raw_config(path: str | Path) -> Dict[str, Any]:
    """Load a config file (toml/json/yaml).

    Returns a raw dict. Validation happens separately.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower().lstrip(".")
    if ext in {"toml"}:
        return _load_toml(p)
    if ext in {"json"}:
        return _load_json(p)
    if ext in {"yaml", "yml"}:
        return _load_yaml(p)

    raise ValueError(f"Unsupported config extension: .{ext} (use .toml/.json/.yaml)")


def _dig(d: Dict[str, Any], key: str, default: Any) -> Any:
    val = d.get(key, default)
    return default if val is None else val


def normalize_app_config(raw: Dict[str, Any]) -> AppConfig:
    """Normalize user config into the minimal shape the CLI expects."""
    npc = raw.get("npc") or {}
    engine = raw.get("engine") or raw.get("runtime") or {}
    session = raw.get("session") or {}
    debug = raw.get("debug") or {}
    tools = raw.get("tools") or {}

    if not isinstance(npc, dict) or not isinstance(engine, dict) or not isinstance(session, dict) or not isinstance(debug, dict) or not isinstance(tools, dict):
        raise ValueError("Config sections npc/engine/session/debug must be objects")

    npc_dir = _dig(npc, "dir", "")
    if not isinstance(npc_dir, str) or not npc_dir.strip():
        raise ValueError("npc.dir is required")

    # Allow debug fields to live under [debug] but be consumed by SessionConfig.
    if debug:
        session = dict(session)
        session.setdefault("debug_dump_dir", _dig(debug, "dump_dir", ".npc/debug"))
        session.setdefault("debug_dump_messages_json", bool(_dig(debug, "dump_messages_json", False)))
        session.setdefault("debug_dump_messages_txt", bool(_dig(debug, "dump_messages_txt", False)))

    return AppConfig(npc_dir=npc_dir, engine=engine, session=session, tools=tools)


def load_app_config(path: str | Path) -> AppConfig:
    return normalize_app_config(load_raw_config(path))