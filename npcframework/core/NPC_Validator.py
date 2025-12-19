from __future__ import annotations

"""
NPCFramework - NPC Directory Validator

Supports:
- Spec-style NPC directories (SPEC.md): npc.yaml + identity/persona/policy required, others optional
- Legacy manifest-style npc.yaml (files/data): for backwards compatibility

Primary:
- validate_npc_dir(npc_dir: str | Path, strict: bool = True) -> ValidationReport

Convenience (for CLI):
- validate_npc_dir_tuple(...) -> tuple[bool, Optional[str]]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # PyYAML
except ImportError as e:
    raise RuntimeError("Missing dependency: PyYAML. Install with: pip install pyyaml") from e


# =============================================================================
# CONFIG
# =============================================================================

META_KEY = "meta"
DB_MISSING_IS_WARNING = True
DEFAULT_STRICT = True

ALLOWED_VERBOSITY = {"low", "medium", "high"}

# --- Spec required files (SPEC.md v0.1)
SPEC_REQUIRED_FILES = ("npc.yaml", "identity.yaml", "persona.yaml", "policy.yaml")
SPEC_OPTIONAL_FILES = ("memory.yaml", "tools.yaml", "kernel.yaml", "actions.yaml", "state.yaml", "perception.yaml", "npc.db")

# --- Legacy manifest schema keys (your older format)
LEGACY_REQUIRED_MANIFEST_KEYS = {"npc_version", "id", "display_name", "files", "data"}
LEGACY_REQUIRED_FILES_KEYS = {"identity", "persona", "policy", "memory", "kernel", "actions", "tools"}
LEGACY_REQUIRED_DATA_KEYS = {"db"}

# Allowed keys per document (STRICT MODE)
# NOTE: Keep this permissive enough to not brick iteration.
ALLOWED_KEYS: Dict[str, set[str]] = {
    "npc.yaml": {"npc_version", "name", "id", "display_name", "description", "author", "tags", "created_at", "updated_at",
                "files", "data", META_KEY},

    "identity.yaml": {"archetype", "description", "core_values", "purpose", "boundaries", "canon", "speech_rules", META_KEY},
    "persona.yaml": {"tone", "style", "verbosity", "humor", "speech_rules", "taboos", "example_lines", "do", "dont", META_KEY},
    "policy.yaml": {"rules", "tool_rules", "boundaries", "refusal_policy", "truthfulness", "refusal_style", META_KEY},

    "memory.yaml": {"seeds", "preferences", "relationships", "facts", "policy", "stores", META_KEY},
    "tools.yaml": {"tools", "enabled", "registry", META_KEY},

    "kernel.yaml": {"strategy", "priorities", "routing", META_KEY},
    "actions.yaml": {"output_modes", "allowed_actions", "formatting", META_KEY},
    "state.yaml": {"state", META_KEY},
    "perception.yaml": {"facts", META_KEY},
}

# Minimal required keys for certain docs (only when the doc exists)
REQUIRED_KEYS: Dict[str, set[str]] = {
    "identity.yaml": {"archetype", "description"},
    "persona.yaml": {"verbosity"},  # tone/style optional, but verbosity we keep as a knob
    "policy.yaml": set(),           # allow you to start minimal; policy can be empty-ish early
}


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class ValidationIssue:
    code: str
    message: str
    path: Optional[str] = None
    severity: str = "ERROR"  # "ERROR" | "WARN"


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
            return f"OK: {self.npc_id or 'unknown'} ({self.display_name or 'unknown'}) v{self.npc_version or 'unknown'} | warnings={self.warn_count}"
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
# YAML UTILITIES
# =============================================================================

def load_yaml(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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
    if not parts:
        return file_name
    return f"{file_name}:" + ".".join(parts)


def _is_str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def _validate_meta(report: ValidationReport, file_name: str, doc: Dict[str, Any]) -> None:
    if META_KEY in doc and doc[META_KEY] is not None and not isinstance(doc[META_KEY], dict):
        report.add("META_TYPE", "meta must be an object/mapping.", _keypath(file_name, META_KEY))


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
    for k in doc.keys():
        if k == META_KEY:
            continue
        if k not in allowed:
            report.add("UNKNOWN_KEY", f"Unknown key '{k}' in {file_name}.", _keypath(file_name, k), severity="WARN")


def _ensure_yaml(report: ValidationReport, path: Path, *, strict: bool) -> Optional[Dict[str, Any]]:
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
# MODE DETECTION
# =============================================================================

def _is_legacy_manifest(manifest: Dict[str, Any]) -> bool:
    return isinstance(manifest.get("files"), dict) and isinstance(manifest.get("data"), dict) and "display_name" in manifest


# =============================================================================
# SPEC-STYLE VALIDATION (SPEC.md)
# =============================================================================

def _validate_spec_style(report: ValidationReport, npc_path: Path, *, strict: bool) -> None:
    # required files
    for fn in SPEC_REQUIRED_FILES:
        p = npc_path / fn
        if not p.exists():
            report.add("FILE_MISSING", f"Missing required file: {fn}", str(p))

    # load npc.yaml (required)
    npc_yaml_path = npc_path / "npc.yaml"
    npc_manifest = _ensure_yaml(report, npc_yaml_path, strict=strict)
    if not npc_manifest:
        return

    # required keys per SPEC.md
    npc_version = npc_manifest.get("npc_version")
    name = npc_manifest.get("name")  # preferred
    legacy_display_name = npc_manifest.get("display_name")  # tolerated
    npc_id = npc_manifest.get("id")

    if not isinstance(npc_version, str) or not npc_version.strip():
        report.add("MANIFEST_TYPE", "npc_version must be a non-empty string.", _keypath("npc.yaml", "npc_version"))

    display_name: Optional[str] = None
    if isinstance(name, str) and name.strip():
        display_name = name.strip()
    elif isinstance(legacy_display_name, str) and legacy_display_name.strip():
        display_name = legacy_display_name.strip()
        report.add("MANIFEST_COMPAT", "npc.yaml uses display_name; prefer 'name' per SPEC.md.", _keypath("npc.yaml", "display_name"), severity="WARN")
    else:
        report.add("MANIFEST_TYPE", "npc.yaml must include non-empty 'name' (or legacy 'display_name').", _keypath("npc.yaml", "name"))

    if npc_id is not None and (not isinstance(npc_id, str) or not npc_id.strip()):
        report.add("MANIFEST_TYPE", "id must be a non-empty string if provided.", _keypath("npc.yaml", "id"))

    report.npc_version = npc_version if isinstance(npc_version, str) else None
    report.display_name = display_name
    report.npc_id = npc_id if isinstance(npc_id, str) else (npc_path.stem if npc_path.suffix == ".npc" else npc_path.name)

    # validate required YAML docs
    identity = _ensure_yaml(report, npc_path / "identity.yaml", strict=strict)
    persona = _ensure_yaml(report, npc_path / "persona.yaml", strict=strict)
    policy = _ensure_yaml(report, npc_path / "policy.yaml", strict=strict)

    if identity:
        if not isinstance(identity.get("archetype"), str):
            report.add("IDENTITY_TYPE", "archetype must be a string.", _keypath("identity.yaml", "archetype"))
        if not isinstance(identity.get("description"), str):
            report.add("IDENTITY_TYPE", "description must be a string.", _keypath("identity.yaml", "description"))
        if "core_values" in identity and identity["core_values"] is not None and not _is_str_list(identity["core_values"]):
            report.add("IDENTITY_TYPE", "core_values must be a list of strings.", _keypath("identity.yaml", "core_values"))
        if "purpose" in identity and identity["purpose"] is not None and not _is_str_list(identity["purpose"]):
            report.add("IDENTITY_TYPE", "purpose must be a list of strings.", _keypath("identity.yaml", "purpose"))

    if persona:
        verbosity = persona.get("verbosity")
        if verbosity is not None:
            if not isinstance(verbosity, str) or verbosity not in ALLOWED_VERBOSITY:
                report.add("PERSONA_ENUM", f"verbosity must be one of {sorted(ALLOWED_VERBOSITY)}.", _keypath("persona.yaml", "verbosity"))
        for k in ("tone", "style", "humor"):
            if k in persona and persona[k] is not None and not isinstance(persona[k], str):
                report.add("PERSONA_TYPE", f"{k} must be a string.", _keypath("persona.yaml", k))
        for list_key in ("speech_rules", "taboos", "example_lines", "do", "dont"):
            if list_key in persona and persona[list_key] is not None and not _is_str_list(persona[list_key]):
                report.add("PERSONA_TYPE", f"{list_key} must be a list of strings.", _keypath("persona.yaml", list_key))

    if policy:
        # policy in SPEC.md is flexible; allow either 'rules' or legacy 'boundaries'
        if "rules" in policy and policy["rules"] is not None and not _is_str_list(policy["rules"]):
            report.add("POLICY_TYPE", "rules must be a list of strings.", _keypath("policy.yaml", "rules"))
        if "tool_rules" in policy and policy["tool_rules"] is not None and not _is_str_list(policy["tool_rules"]):
            report.add("POLICY_TYPE", "tool_rules must be a list of strings.", _keypath("policy.yaml", "tool_rules"))
        if "boundaries" in policy and policy["boundaries"] is not None and not _is_str_list(policy["boundaries"]):
            report.add("POLICY_TYPE", "boundaries must be a list of strings.", _keypath("policy.yaml", "boundaries"))

    # optional docs: validate if present
    for opt in ("memory.yaml", "tools.yaml", "kernel.yaml", "actions.yaml", "state.yaml", "perception.yaml"):
        p = npc_path / opt
        if p.exists():
            _ensure_yaml(report, p, strict=strict)

    # db file: warn-only
    db_path = npc_path / "npc.db"
    if not db_path.exists():
        if DB_MISSING_IS_WARNING:
            report.add("DB_MISSING", "Database file does not exist yet (ok for first run).", str(db_path), severity="WARN")
        else:
            report.add("DB_MISSING", "Database file does not exist yet.", str(db_path))


# =============================================================================
# LEGACY MANIFEST VALIDATION (your old format)
# =============================================================================

def _validate_legacy_manifest(report: ValidationReport, npc_path: Path, manifest: Dict[str, Any], *, strict: bool) -> None:
    _validate_required_keys(report, "npc.yaml", manifest, LEGACY_REQUIRED_MANIFEST_KEYS)
    if strict and "npc.yaml" in ALLOWED_KEYS:
        _validate_allowed_keys(report, "npc.yaml", manifest, ALLOWED_KEYS["npc.yaml"])
    _validate_meta(report, "npc.yaml", manifest)

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
        return
    if not isinstance(data, dict):
        report.add("MANIFEST_TYPE", "data must be a mapping/object.", _keypath("npc.yaml", "data"))
        return

    _validate_required_keys(report, "npc.yaml", files, LEGACY_REQUIRED_FILES_KEYS, prefix=("files",))
    _validate_required_keys(report, "npc.yaml", data, LEGACY_REQUIRED_DATA_KEYS, prefix=("data",))

    # resolve referenced file paths
    resolved: Dict[str, Path] = {}
    for key in LEGACY_REQUIRED_FILES_KEYS:
        rel = files.get(key)
        if not isinstance(rel, str) or not rel.strip():
            report.add("MANIFEST_TYPE", f"files.{key} must be a non-empty string path.", _keypath("npc.yaml", "files", key))
            continue
        resolved[key] = npc_path / rel

    # db path (warn-only if missing)
    db_rel = data.get("db")
    if isinstance(db_rel, str) and db_rel.strip():
        db_path = npc_path / db_rel
        if not db_path.exists():
            sev = "WARN" if DB_MISSING_IS_WARNING else "ERROR"
            report.add("DB_MISSING", "Database file does not exist yet (ok for first run).", str(db_path), severity=sev)
    else:
        report.add("MANIFEST_TYPE", "data.db must be a non-empty string path.", _keypath("npc.yaml", "data", "db"))

    # validate referenced yaml docs (if present, validate keys/types lightly)
    for legacy_key in LEGACY_REQUIRED_FILES_KEYS:
        p = resolved.get(legacy_key)
        if not p:
            continue
        _ensure_yaml(report, p, strict=strict)


# =============================================================================
# PUBLIC ENTRYPOINTS
# =============================================================================

def validate_npc_dir(npc_dir: str | Path, *, strict: bool = DEFAULT_STRICT) -> ValidationReport:
    report = ValidationReport(ok=True)
    npc_path = Path(npc_dir)

    if not npc_path.exists():
        report.add("NPC_DIR_NOT_FOUND", "NPC directory does not exist.", str(npc_path))
        return report
    if not npc_path.is_dir():
        report.add("NPC_NOT_A_DIR", "Expected NPC path to be a directory.", str(npc_path))
        return report

    manifest_path = npc_path / "npc.yaml"
    if not manifest_path.exists():
        report.add("MANIFEST_MISSING", "Missing npc.yaml manifest.", str(manifest_path))
        return report

    manifest, err = load_yaml(manifest_path)
    if err or not manifest:
        report.add("MANIFEST_INVALID", err or "Invalid npc.yaml", str(manifest_path))
        return report

    # detect mode
    if _is_legacy_manifest(manifest):
        _validate_legacy_manifest(report, npc_path, manifest, strict=strict)
    else:
        _validate_spec_style(report, npc_path, strict=strict)

    return report


def validate_npc_dir_tuple(npc_dir: str | Path, *, strict: bool = DEFAULT_STRICT) -> Tuple[bool, Optional[str]]:
    """
    Convenience adapter for CLI / older code.
    Returns (ok, error_message). If ok=True, error_message=None.
    """
    rep = validate_npc_dir(npc_dir, strict=strict)
    if rep.ok:
        return True, None
    # summarize errors
    msgs = []
    for issue in rep.issues:
        if issue.severity.upper() == "ERROR":
            loc = f" [{issue.path}]" if issue.path else ""
            msgs.append(f"{issue.code}{loc}: {issue.message}")
    return False, "\n".join(msgs) if msgs else "Validation failed."


# =============================================================================
# CLI HELPER (optional)
# =============================================================================

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate an NPC directory.")
    parser.add_argument("npc_dir", help="Path to npc directory (e.g., npc/kevin.npc)")
    parser.add_argument("--non-strict", action="store_true", help="Allow unknown keys (unknown keys become WARN or ignored).")
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
