from __future__ import annotations

"""
NPCFramework - NPC Directory Validator (v0.1)

PURPOSE
- Validate an NPC directory structure and its YAML documents.
- Enforce minimal required keys + optional strict unknown-key policy.
- Validate basic types/enums to catch schema drift early.

PRIMARY ENTRYPOINT
- validate_npc_dir(npc_dir: str | Path, strict: bool = True) -> ValidationReport

I/O SHAPE
Input:
- npc_dir: path to an *.npc directory containing npc.yaml and referenced YAML files

Output:
- ValidationReport:
    - ok: bool
    - npc_id, npc_version, display_name: optional str
    - issues: list[ValidationIssue]
    - counts: error/warn counts + helpers for printing/debugging

NOTES
- This validator is intentionally "schema-light" and "debug-heavy".
- It should help you iterate fast, not punish you for first-run realities (like missing DB).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # PyYAML
except ImportError as e:
    raise RuntimeError("Missing dependency: PyYAML. Install with: pip install pyyaml") from e


# =============================================================================
# CONFIG (debug knobs live here)
# =============================================================================

META_KEY = "meta"

# If True, missing DB file is a WARNING (non-fatal), which is correct for first run.
DB_MISSING_IS_WARNING = True

# Enum constraints
ALLOWED_VERBOSITY = {"low", "medium", "high"}
ALLOWED_PRIVACY_MODE = {"user_owned", "shared", "none"}

# Default strict mode behavior:
# - Unknown keys flagged (except meta)
# - Types/enums validated
DEFAULT_STRICT = True


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class ValidationIssue:
    code: str
    message: str
    path: Optional[str] = None      # file path or key path
    severity: str = "ERROR"         # "ERROR" | "WARN"


@dataclass
class ValidationReport:
    ok: bool
    npc_id: Optional[str] = None
    npc_version: Optional[str] = None
    display_name: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, code: str, message: str, path: Optional[str] = None, *, severity: str = "ERROR") -> None:
        self.issues.append(ValidationIssue(code=code, message=message, path=path, severity=severity))
        if severity.upper() == "ERROR":
            self.ok = False

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity.upper() == "ERROR")

    @property
    def warn_count(self) -> int:
        return sum(1 for i in self.issues if i.severity.upper() == "WARN")

    def summary(self) -> str:
        if self.ok:
            return f"OK: {self.npc_id} ({self.display_name}) v{self.npc_version} | warnings={self.warn_count}"
        return f"INVALID: {self.npc_id or 'unknown'} | errors={self.error_count}, warnings={self.warn_count}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "npc_id": self.npc_id,
            "npc_version": self.npc_version,
            "display_name": self.display_name,
            "errors": self.error_count,
            "warnings": self.warn_count,
            "issues": [
                {"code": x.code, "severity": x.severity, "path": x.path, "message": x.message}
                for x in self.issues
            ],
        }


# =============================================================================
# SCHEMA DEFINITIONS (v0.1)
# =============================================================================

REQUIRED_MANIFEST_KEYS = {"npc_version", "id", "display_name", "files", "data"}
REQUIRED_FILES_KEYS = {"identity", "persona", "policy", "memory", "kernel", "actions", "tools"}
REQUIRED_DATA_KEYS = {"db"}

# Allowed keys per document (STRICT MODE)
# meta is allowed everywhere as a "compat bucket".
ALLOWED_KEYS: Dict[str, set[str]] = {
    "npc.yaml": {"npc_version", "id", "display_name", "files", "data", META_KEY},

    "identity.yaml": {"archetype", "description", "core_values", "purpose", META_KEY},
    "persona.yaml": {"tone", "style", "verbosity", "humor", "speech_rules", "taboos", "example_lines", META_KEY},
    "policy.yaml": {"boundaries", "refusal_policy", "truthfulness", META_KEY},
    "memory.yaml": {"policy", "stores", "seeds", META_KEY},
    "kernel.yaml": {"strategy", "priorities", "routing", META_KEY},
    "actions.yaml": {"output_modes", "allowed_actions", "formatting", META_KEY},
    "tools.yaml": {"enabled", "registry", META_KEY},
}

# Minimal required keys per referenced YAML file
REQUIRED_KEYS: Dict[str, set[str]] = {
    "identity.yaml": {"archetype", "description", "core_values", "purpose"},
    "persona.yaml": {"tone", "style", "verbosity"},
    "policy.yaml": {"boundaries"},
    "memory.yaml": {"policy", "stores"},
    "kernel.yaml": {"strategy", "priorities", "routing"},
    "actions.yaml": {"output_modes", "allowed_actions", "formatting"},
    "tools.yaml": {"enabled", "registry"},
}


# =============================================================================
# YAML UTILITIES
# =============================================================================

def load_yaml(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Returns (data, error_message). error_message is None when successful."""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        return None, f"Failed to read file: {e}"

    try:
        data = yaml.safe_load(raw)
    except Exception as e:
        return None, f"YAML parse error: {e}"

    if data is None:
        return None, "YAML file is empty (expected a mapping/object)."
    if not isinstance(data, dict):
        return None, f"Expected YAML mapping/object at root, got: {type(data).__name__}"

    return data, None


def _keypath(file_name: str, *parts: str) -> str:
    """Human-greppable key path format: file.yaml:key.subkey.subkey"""
    if not parts:
        return file_name
    return f"{file_name}:" + ".".join(parts)


def _is_str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def _validate_meta(report: ValidationReport, file_name: str, doc: Dict[str, Any]) -> None:
    """meta is allowed everywhere but must be an object if present."""
    if META_KEY in doc and doc[META_KEY] is not None and not isinstance(doc[META_KEY], dict):
        report.add("META_TYPE", "meta must be an object/mapping.", _keypath(file_name, META_KEY))


# =============================================================================
# GENERIC KEY CHECKS
# =============================================================================

def _validate_required_keys(
    report: ValidationReport,
    file_name: str,
    doc: Dict[str, Any],
    required: set[str],
    prefix: Tuple[str, ...] = (),
) -> None:
    for k in required:
        if k not in doc:
            report.add("MISSING_KEY", f"Missing key: {'.'.join(prefix + (k,))}", _keypath(file_name, *prefix))


def _validate_allowed_keys(report: ValidationReport, file_name: str, doc: Dict[str, Any], allowed: set[str]) -> None:
    """
    Strict unknown-key policy:
    - only keys in `allowed` are allowed
    - meta is always allowed (bulletproof)
    """
    for k in doc.keys():
        if k == META_KEY:
            continue
        if k not in allowed:
            report.add("UNKNOWN_KEY", f"Unknown key '{k}' in {file_name}.", _keypath(file_name, k))


# =============================================================================
# FILE LOADER + BASE CHECKS
# =============================================================================

def _ensure_file(
    report: ValidationReport,
    path: Optional[Path],
    label: str,
    *,
    strict: bool,
) -> Optional[Dict[str, Any]]:
    """
    Loads YAML + applies:
    - existence checks
    - YAML parse checks
    - strict unknown-key checks (if strict)
    - required-key checks (if configured)
    - meta type check
    """
    if path is None:
        report.add("FILE_REF_MISSING", f"Manifest missing reference for {label}.", label)
        return None

    if not path.exists():
        report.add("FILE_MISSING", f"Missing file: {path.name}", str(path))
        return None

    doc, err = load_yaml(path)
    if err:
        report.add("FILE_INVALID", err, str(path))
        return None

    allowed = ALLOWED_KEYS.get(path.name)
    if strict and allowed:
        _validate_allowed_keys(report, path.name, doc, allowed)

    _validate_meta(report, path.name, doc)

    req = REQUIRED_KEYS.get(path.name)
    if req:
        _validate_required_keys(report, path.name, doc, req)

    return doc


# =============================================================================
# CORE VALIDATION
# =============================================================================

def validate_npc_dir(npc_dir: str | Path, *, strict: bool = DEFAULT_STRICT) -> ValidationReport:
    """
    Validate an NPC directory (e.g. npc/kevin.npc).

    strict=True:
      - flags unknown keys in YAML files (except 'meta')
      - validates types/enums where defined
    """
    report = ValidationReport(ok=True)

    npc_path = Path(npc_dir)

    # --- path sanity
    if not npc_path.exists():
        report.add("NPC_DIR_NOT_FOUND", "NPC directory does not exist.", str(npc_path))
        return report
    if not npc_path.is_dir():
        report.add("NPC_NOT_A_DIR", "Expected NPC path to be a directory.", str(npc_path))
        return report

    # --- manifest load
    manifest_path = npc_path / "npc.yaml"
    if not manifest_path.exists():
        report.add("MANIFEST_MISSING", "Missing npc.yaml manifest.", str(manifest_path))
        return report

    manifest, err = load_yaml(manifest_path)
    if err:
        report.add("MANIFEST_INVALID", err, str(manifest_path))
        return report

    _validate_required_keys(report, "npc.yaml", manifest, REQUIRED_MANIFEST_KEYS)
    if strict:
        _validate_allowed_keys(report, "npc.yaml", manifest, ALLOWED_KEYS["npc.yaml"])
    _validate_meta(report, "npc.yaml", manifest)

    # --- manifest types
    npc_version = manifest.get("npc_version")
    npc_id = manifest.get("id")
    display_name = manifest.get("display_name")

    if not isinstance(npc_version, str):
        report.add("MANIFEST_TYPE", "npc_version must be a string.", _keypath("npc.yaml", "npc_version"))
    if not isinstance(npc_id, str) or not npc_id.strip():
        report.add("MANIFEST_TYPE", "id must be a non-empty string.", _keypath("npc.yaml", "id"))
    if not isinstance(display_name, str) or not display_name.strip():
        report.add("MANIFEST_TYPE", "display_name must be a non-empty string.", _keypath("npc.yaml", "display_name"))

    report.npc_version = npc_version if isinstance(npc_version, str) else None
    report.npc_id = npc_id if isinstance(npc_id, str) else None
    report.display_name = display_name if isinstance(display_name, str) else None

    files = manifest.get("files")
    data = manifest.get("data")

    if not isinstance(files, dict):
        report.add("MANIFEST_TYPE", "files must be a mapping/object.", _keypath("npc.yaml", "files"))
        return report
    if not isinstance(data, dict):
        report.add("MANIFEST_TYPE", "data must be a mapping/object.", _keypath("npc.yaml", "data"))
        return report

    _validate_required_keys(report, "npc.yaml", files, REQUIRED_FILES_KEYS, prefix=("files",))
    _validate_required_keys(report, "npc.yaml", data, REQUIRED_DATA_KEYS, prefix=("data",))

    # --- resolve file paths referenced by manifest
    resolved: Dict[str, Path] = {}
    for key in REQUIRED_FILES_KEYS:
        rel = files.get(key)
        if not isinstance(rel, str) or not rel.strip():
            report.add("MANIFEST_TYPE", f"files.{key} must be a non-empty string path.", _keypath("npc.yaml", "files", key))
            continue
        resolved[key] = npc_path / rel

    # --- db path (warn-only if missing)
    db_rel = data.get("db")
    if isinstance(db_rel, str) and db_rel.strip():
        db_path = npc_path / db_rel
        if not db_path.exists():
            if DB_MISSING_IS_WARNING:
                report.add("DB_MISSING", "Database file does not exist yet (ok for first run).", str(db_path), severity="WARN")
            else:
                report.add("DB_MISSING", "Database file does not exist yet.", str(db_path))
    else:
        report.add("MANIFEST_TYPE", "data.db must be a non-empty string path.", _keypath("npc.yaml", "data", "db"))

    # --- validate referenced yaml docs
    _validate_identity(report, resolved.get("identity"), strict=strict)
    _validate_persona(report, resolved.get("persona"), strict=strict)
    _validate_policy(report, resolved.get("policy"), strict=strict)
    _validate_memory(report, resolved.get("memory"), strict=strict)
    _validate_kernel(report, resolved.get("kernel"), strict=strict)
    _validate_actions(report, resolved.get("actions"), strict=strict)
    _validate_tools(report, resolved.get("tools"), strict=strict)

    return report


# =============================================================================
# FILE-SPECIFIC VALIDATORS
# =============================================================================

def _validate_identity(report: ValidationReport, path: Optional[Path], *, strict: bool) -> None:
    doc = _ensure_file(report, path, "identity", strict=strict)
    if not doc:
        return

    if not isinstance(doc.get("archetype"), str):
        report.add("IDENTITY_TYPE", "archetype must be a string.", _keypath("identity.yaml", "archetype"))

    if not isinstance(doc.get("description"), str):
        report.add("IDENTITY_TYPE", "description must be a string.", _keypath("identity.yaml", "description"))

    if not _is_str_list(doc.get("core_values")):
        report.add("IDENTITY_TYPE", "core_values must be a list of strings.", _keypath("identity.yaml", "core_values"))

    if not _is_str_list(doc.get("purpose")):
        report.add("IDENTITY_TYPE", "purpose must be a list of strings.", _keypath("identity.yaml", "purpose"))


def _validate_persona(report: ValidationReport, path: Optional[Path], *, strict: bool) -> None:
    doc = _ensure_file(report, path, "persona", strict=strict)
    if not doc:
        return

    for k in ("tone", "style", "humor"):
        if k in doc and doc[k] is not None and not isinstance(doc[k], str):
            report.add("PERSONA_TYPE", f"{k} must be a string.", _keypath("persona.yaml", k))

    verbosity = doc.get("verbosity")
    if not isinstance(verbosity, str) or verbosity not in ALLOWED_VERBOSITY:
        report.add("PERSONA_ENUM", f"verbosity must be one of {sorted(ALLOWED_VERBOSITY)}.", _keypath("persona.yaml", "verbosity"))

    for list_key in ("speech_rules", "taboos", "example_lines"):
        if list_key in doc and doc[list_key] is not None and not _is_str_list(doc[list_key]):
            report.add("PERSONA_TYPE", f"{list_key} must be a list of strings.", _keypath("persona.yaml", list_key))


def _validate_policy(report: ValidationReport, path: Optional[Path], *, strict: bool) -> None:
    doc = _ensure_file(report, path, "policy", strict=strict)
    if not doc:
        return

    if not _is_str_list(doc.get("boundaries")):
        report.add("POLICY_TYPE", "boundaries must be a list of strings.", _keypath("policy.yaml", "boundaries"))

    for list_key in ("refusal_policy", "truthfulness"):
        if list_key in doc and doc[list_key] is not None and not _is_str_list(doc[list_key]):
            report.add("POLICY_TYPE", f"{list_key} must be a list of strings.", _keypath("policy.yaml", list_key))


def _validate_memory(report: ValidationReport, path: Optional[Path], *, strict: bool) -> None:
    doc = _ensure_file(report, path, "memory", strict=strict)
    if not doc:
        return

    policy = doc.get("policy")
    if not isinstance(policy, dict):
        report.add("MEMORY_TYPE", "policy must be an object/mapping.", _keypath("memory.yaml", "policy"))
        return

    persistent = policy.get("persistent")
    if not isinstance(persistent, bool):
        report.add("MEMORY_TYPE", "policy.persistent must be boolean.", _keypath("memory.yaml", "policy", "persistent"))

    privacy_mode = policy.get("privacy_mode")
    if not isinstance(privacy_mode, str) or privacy_mode not in ALLOWED_PRIVACY_MODE:
        report.add("MEMORY_ENUM", f"policy.privacy_mode must be one of {sorted(ALLOWED_PRIVACY_MODE)}.", _keypath("memory.yaml", "policy", "privacy_mode"))

    for rk in ("write_rules", "forget_rules"):
        if rk in policy and policy[rk] is not None and not _is_str_list(policy[rk]):
            report.add("MEMORY_TYPE", f"policy.{rk} must be a list of strings.", _keypath("memory.yaml", "policy", rk))

    stores = doc.get("stores")
    if not isinstance(stores, dict):
        report.add("MEMORY_TYPE", "stores must be an object/mapping.", _keypath("memory.yaml", "stores"))
    else:
        for store_name, store_cfg in stores.items():
            if not isinstance(store_cfg, dict):
                report.add("MEMORY_TYPE", f"stores.{store_name} must be an object.", _keypath("memory.yaml", "stores", str(store_name)))
                continue
            retrieval = store_cfg.get("retrieval")
            if retrieval is not None and not isinstance(retrieval, str):
                report.add("MEMORY_TYPE", f"stores.{store_name}.retrieval must be a string.", _keypath("memory.yaml", "stores", str(store_name), "retrieval"))

    seeds = doc.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, dict):
            report.add("MEMORY_TYPE", "seeds must be an object/mapping.", _keypath("memory.yaml", "seeds"))
        else:
            semantic = seeds.get("semantic")
            if semantic is not None:
                if not isinstance(semantic, list) or not all(isinstance(x, dict) for x in semantic):
                    report.add("MEMORY_TYPE", "seeds.semantic must be a list of objects.", _keypath("memory.yaml", "seeds", "semantic"))
                else:
                    for i, item in enumerate(semantic):
                        if "key" not in item or "value" not in item:
                            report.add("MEMORY_SEED", "Each seeds.semantic item must have key and value.", _keypath("memory.yaml", "seeds", "semantic", str(i)))


def _validate_kernel(report: ValidationReport, path: Optional[Path], *, strict: bool) -> None:
    doc = _ensure_file(report, path, "kernel", strict=strict)
    if not doc:
        return

    if not isinstance(doc.get("strategy"), str):
        report.add("KERNEL_TYPE", "strategy must be a string.", _keypath("kernel.yaml", "strategy"))

    if not _is_str_list(doc.get("priorities")):
        report.add("KERNEL_TYPE", "priorities must be a list of strings.", _keypath("kernel.yaml", "priorities"))

    routing = doc.get("routing")
    if not isinstance(routing, dict):
        report.add("KERNEL_TYPE", "routing must be an object/mapping.", _keypath("kernel.yaml", "routing"))
        return

    for sys_name in ("system1", "system2", "system3"):
        if sys_name not in routing:
            report.add("KERNEL_MISSING", f"routing must include {sys_name}.", _keypath("kernel.yaml", "routing", sys_name))
            continue
        if not isinstance(routing[sys_name], dict):
            report.add("KERNEL_TYPE", f"routing.{sys_name} must be an object.", _keypath("kernel.yaml", "routing", sys_name))


def _validate_actions(report: ValidationReport, path: Optional[Path], *, strict: bool) -> None:
    doc = _ensure_file(report, path, "actions", strict=strict)
    if not doc:
        return

    output_modes = doc.get("output_modes")
    if not isinstance(output_modes, list) or not all(isinstance(x, str) for x in output_modes):
        report.add("ACTIONS_TYPE", "output_modes must be a list of strings.", _keypath("actions.yaml", "output_modes"))

    allowed_actions = doc.get("allowed_actions")
    if not isinstance(allowed_actions, list) or not all(isinstance(x, dict) for x in allowed_actions):
        report.add("ACTIONS_TYPE", "allowed_actions must be a list of objects.", _keypath("actions.yaml", "allowed_actions"))
    else:
        for i, act in enumerate(allowed_actions):
            if not isinstance(act.get("name"), str):
                report.add("ACTIONS_TYPE", "allowed_actions[].name must be a string.", _keypath("actions.yaml", "allowed_actions", str(i), "name"))
            if "description" in act and act["description"] is not None and not isinstance(act["description"], str):
                report.add("ACTIONS_TYPE", "allowed_actions[].description must be a string.", _keypath("actions.yaml", "allowed_actions", str(i), "description"))

    fmt = doc.get("formatting")
    if not isinstance(fmt, dict):
        report.add("ACTIONS_TYPE", "formatting must be an object/mapping.", _keypath("actions.yaml", "formatting"))
    else:
        for lk in ("prefer", "avoid"):
            if lk in fmt and fmt[lk] is not None and not _is_str_list(fmt[lk]):
                report.add("ACTIONS_TYPE", f"formatting.{lk} must be a list of strings.", _keypath("actions.yaml", "formatting", lk))


def _validate_tools(report: ValidationReport, path: Optional[Path], *, strict: bool) -> None:
    doc = _ensure_file(report, path, "tools", strict=strict)
    if not doc:
        return

    enabled = doc.get("enabled")
    if not isinstance(enabled, bool):
        report.add("TOOLS_TYPE", "enabled must be boolean.", _keypath("tools.yaml", "enabled"))

    registry = doc.get("registry")
    if not isinstance(registry, list):
        report.add("TOOLS_TYPE", "registry must be a list.", _keypath("tools.yaml", "registry"))
    else:
        for i, item in enumerate(registry):
            if not isinstance(item, (str, dict)):
                report.add("TOOLS_TYPE", "registry items must be strings or objects.", _keypath("tools.yaml", "registry", str(i)))


# =============================================================================
# CLI HELPER (optional)
# =============================================================================

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate an NPC directory (NPCFramework v0.1).")
    parser.add_argument("npc_dir", help="Path to npc directory (e.g., npc/kevin.npc)")
    parser.add_argument("--non-strict", action="store_true", help="Allow unknown keys (except required key checks still apply).")
    args = parser.parse_args()

    report = validate_npc_dir(args.npc_dir, strict=not args.non_strict)

    if report.ok:
        print(f"✅ {report.summary()}")
        if report.warn_count:
            for issue in report.issues:
                if issue.severity.upper() == "WARN":
                    loc = f" [{issue.path}]" if issue.path else ""
                    print(f"- {issue.severity} {issue.code}{loc}: {issue.message}")
        sys.exit(0)

    print(f"❌ {report.summary()}")
    for issue in report.issues:
        loc = f" [{issue.path}]" if issue.path else ""
        print(f"- {issue.severity} {issue.code}{loc}: {issue.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()
