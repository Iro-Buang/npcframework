from __future__ import annotations

"""
Runtime_Config

Single place to load framework-level runtime configs from the repo-level /configs folder.

Why:
- Keeps "constants" out of code (you said YAML or we riot).
- Avoids circular imports between orchestrator / prompt compiler.
- Provides safe defaults when configs are missing.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

_CACHE: Dict[str, Dict[str, Any]] = {}
_DEBUG_CACHE: Optional[Dict[str, Any]] = None
_MISSING_KEYS_LOGGED: set[str] = set()


def _find_configs_dir(start: Optional[Path] = None) -> Optional[Path]:
    """
    Heuristic search for a 'configs' directory.
    Search order:
      1) start (or CWD) and its parents
      2) this file's parents
    """
    roots = []
    roots.append((start or Path.cwd()).resolve())
    roots.append(Path(__file__).resolve())

    for root in roots:
        for p in [root] + list(root.parents):
            c = p / "configs"
            if c.exists() and c.is_dir():
                return c
    return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        # YAML is optional at install time; default configs will be used.
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_config_yaml(filename: str) -> Dict[str, Any]:
    """
    Load configs/<filename>.yaml into a dict (cached).
    If missing or YAML isn't installed, returns {}.
    """
    key = filename.strip()
    if key in _CACHE:
        return _CACHE[key]

    cfg_dir = _find_configs_dir()
    if not cfg_dir:        # Optional: warn when configs dir can't be located
        if os.environ.get('NPCFW_WARN_MISSING_FILES','').strip().lower() in {'1','true','yes','y','on'}:
            try:
                import sys
                print(f"[npcframework][config] could not locate 'configs' directory (requested {key})", file=sys.stderr)
            except Exception:
                pass
        _CACHE[key] = {}
        return _CACHE[key]

    # allow passing either "foo" or "foo.yaml"
    fn = key if key.lower().endswith((".yaml", ".yml")) else f"{key}.yaml"
    path = cfg_dir / fn
    if not path.exists():
        if os.environ.get('NPCFW_WARN_MISSING_FILES','').strip().lower() in {'1','true','yes','y','on'}:
            try:
                import sys
                print(f"[npcframework][config] missing file configs/{fn} (using defaults)", file=sys.stderr)
            except Exception:
                pass
        _CACHE[key] = {}
        return _CACHE[key]

    _CACHE[key] = _load_yaml(path)
    return _CACHE[key]


def get_str(d: Dict[str, Any], k: str, default: str, *, filename: str = "") -> str:
    if k not in d and filename:
        _warn_missing_key(filename, k)
    v = d.get(k, default)
    return default if v is None else str(v)


def get_int(d: Dict[str, Any], k: str, default: int, *, filename: str = "") -> int:
    if k not in d and filename:
        _warn_missing_key(filename, k)
    v = d.get(k, default)
    try:
        return int(v)
    except Exception:
        return default


def get_bool(d: Dict[str, Any], k: str, default: bool, *, filename: str = "") -> bool:
    if k not in d and filename:
        _warn_missing_key(filename, k)
    v = d.get(k, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def get_list_str(d: Dict[str, Any], k: str, default: list[str], *, filename: str = "") -> list[str]:
    if k not in d and filename:
        _warn_missing_key(filename, k)
    v = d.get(k, None)
    if v is None:
        return list(default)
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    return list(default)


def _debug_cfg() -> Dict[str, Any]:
    """Lazy-load runtime_debug.yaml (if present)."""
    global _DEBUG_CACHE
    if _DEBUG_CACHE is not None:
        return _DEBUG_CACHE
    _DEBUG_CACHE = load_config_yaml("runtime_debug")
    if not isinstance(_DEBUG_CACHE, dict):
        _DEBUG_CACHE = {}
    return _DEBUG_CACHE


def _warn_missing_key(filename: str, key: str) -> None:
    """Print a one-time warning when a config key is missing."""
    dbg = _debug_cfg()
    enabled = get_bool(dbg, "warn_missing_keys", False)
    if not enabled:
        return
    msg = f"[npcframework][config] missing key '{key}' in configs/{filename}.yaml (using default)"
    if msg in _MISSING_KEYS_LOGGED:
        return
    _MISSING_KEYS_LOGGED.add(msg)
    try:
        import sys
        print(msg, file=sys.stderr)
    except Exception:
        pass
