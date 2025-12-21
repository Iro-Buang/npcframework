from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


# -----------------------------
# Types
# -----------------------------

YamlDict = Dict[str, Any]


@dataclass(frozen=True)
class NPCPaths:
    root: Path
    manifest: Path
    db: Path
    files: Dict[str, Path]  # identity/persona/policy/etc -> absolute Path


@dataclass(frozen=True)
class NPCBundle:
    """Loaded NPC config bundle (static config + resolved paths)."""
    manifest: YamlDict
    identity: YamlDict
    persona: YamlDict
    policy: YamlDict
    goals: YamlDict
    memory: YamlDict
    kernel: YamlDict
    actions: YamlDict
    tools: YamlDict
    paths: NPCPaths


# -----------------------------
# Exceptions
# -----------------------------

class NPCLoadError(RuntimeError):
    pass


# -----------------------------
# YAML helpers
# -----------------------------

def _load_yaml(path: Path) -> YamlDict:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        raise NPCLoadError(f"Failed to read YAML file: {path} ({e})") from e

    try:
        data = yaml.safe_load(raw)
    except Exception as e:
        raise NPCLoadError(f"YAML parse error in {path}: {e}") from e

    if data is None:
        raise NPCLoadError(f"YAML file is empty: {path}")
    if not isinstance(data, dict):
        raise NPCLoadError(f"Expected YAML mapping/object at root in {path}, got {type(data).__name__}")

    return data


def _require_key(obj: YamlDict, key: str, *, where: str) -> Any:
    if key not in obj:
        raise NPCLoadError(f"Missing required key '{key}' in {where}")
    return obj[key]


def _require_str(obj: YamlDict, key: str, *, where: str) -> str:
    val = _require_key(obj, key, where=where)
    if not isinstance(val, str) or not val.strip():
        raise NPCLoadError(f"Key '{key}' must be a non-empty string in {where}")
    return val


def _require_dict(obj: YamlDict, key: str, *, where: str) -> YamlDict:
    val = _require_key(obj, key, where=where)
    if not isinstance(val, dict):
        raise NPCLoadError(f"Key '{key}' must be a mapping/object in {where}")
    return val


# -----------------------------
# Loader
# -----------------------------

REQUIRED_DOC_KEYS = ("identity", "persona", "policy", "goals", "memory", "kernel", "actions", "tools")

def extract_existential_goals(goals_yaml: dict) -> list[str]:
    raw = goals_yaml.get("existential", [])
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text", "")
                if isinstance(text, str) and text.strip():
                    out.append(text.strip())
    return out

def parse_existential_goals(goals_yaml: dict) -> list[str]:
    raw = goals_yaml.get("existential", [])
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    out.append(text.strip())
    return out


def load_npc(npc_dir: Union[str, Path]) -> NPCBundle:
    """
    Load an NPC directory (e.g. npc/kevin.npc) into a bundle.

    This loader is intentionally dumb:
    - resolves paths
    - loads YAML docs
    - returns a normalized bundle
    Validation should be done by npcvalidator.py.
    """
    root = Path(npc_dir)

    if not root.exists():
        raise NPCLoadError(f"NPC directory does not exist: {root}")
    if not root.is_dir():
        raise NPCLoadError(f"NPC path is not a directory: {root}")

    manifest_path = root / "npc.yaml"
    if not manifest_path.exists():
        raise NPCLoadError(f"Missing manifest: {manifest_path}")

    manifest = _load_yaml(manifest_path)

    # Basic required manifest keys (light sanity; validator does the heavy lifting)
    _require_str(manifest, "npc_version", where="npc.yaml")
    _require_str(manifest, "id", where="npc.yaml")
    _require_str(manifest, "display_name", where="npc.yaml")

    files = _require_dict(manifest, "files", where="npc.yaml")
    data = _require_dict(manifest, "data", where="npc.yaml")

    # Resolve required document paths
    resolved_files: Dict[str, Path] = {}
    for k in REQUIRED_DOC_KEYS:
        rel = _require_str(files, k, where="npc.yaml:files")
        resolved = (root / rel).resolve()
        resolved_files[k] = resolved
        if not resolved.exists():
            raise NPCLoadError(f"Missing required file for '{k}': {resolved}")

    # Resolve DB path
    db_rel = _require_str(data, "db", where="npc.yaml:data")
    db_path = (root / db_rel).resolve()

    # Load all docs
    identity = _load_yaml(resolved_files["identity"])
    persona = _load_yaml(resolved_files["persona"])
    policy = _load_yaml(resolved_files["policy"])
    goals = _load_yaml(resolved_files["goals"])
    memory = _load_yaml(resolved_files["memory"])
    kernel = _load_yaml(resolved_files["kernel"])
    actions = _load_yaml(resolved_files["actions"])
    tools = _load_yaml(resolved_files["tools"])

    paths = NPCPaths(
        root=root.resolve(),
        manifest=manifest_path.resolve(),
        db=db_path,
        files={k: v for k, v in resolved_files.items()},
    )

    return NPCBundle(
        manifest=manifest,
        identity=identity,
        persona=persona,
        policy=policy,
        goals=goals,
        memory=memory,
        kernel=kernel,
        actions=actions,
        tools=tools,
        paths=paths,
    )


# -----------------------------
# Convenience
# -----------------------------

def load_npc_id(npc_dir: Union[str, Path]) -> str:
    """Fast-path: load only npc.yaml and return id."""
    root = Path(npc_dir)
    manifest = _load_yaml(root / "npc.yaml")
    return _require_str(manifest, "id", where="npc.yaml")


def try_load_npc(npc_dir: Union[str, Path]) -> Optional[NPCBundle]:
    """Returns None instead of raising, for UI-friendly usage."""
    try:
        return load_npc(npc_dir)
    except NPCLoadError:
        return None
