# NPCFramework NPC Spec (v0.1)

This document defines the on-disk format for an NPC directory (`*.npc/`) used by NPCFramework.

NPCFramework is **deterministic-first** and **LLM-optional**. The NPC directory is the portable “soul capsule”: identity + persona + policy + memory seeds + (optional) tools + (optional) event DB.

---

## 0. Terms

- **NPC Directory**: A folder ending in `.npc` containing YAML files and optional SQLite DB.
- **Spec Version**: The version of this NPC directory format (`npc_version` in `npc.yaml`).
- **Runtime**: The application executing turns (prompt compiler, orchestrator, inference backend).
- **Environment/App**: Owns world truth and state; the NPC directory stores only the NPC’s configuration + memory artifacts.

---

## 1. Directory Naming & Layout

### 1.1 Naming
- An NPC directory MUST end with `.npc` (example: `kevin.npc/`).
- Directory name SHOULD be stable (avoid renaming unless you migrate or intentionally fork).

### 1.2 Minimum Layout (required files)
An NPC directory MUST contain:

- `npc.yaml` — metadata + spec version (required)
- `identity.yaml` — stable identity facts (required)
- `persona.yaml` — tone/style/behavior knobs (required)
- `policy.yaml` — hard constraints / safety / behavioral rules (required)

### 1.3 Optional Files
The following files are optional:

- `memory.yaml` — memory seeds, long-term preferences, canonical facts
- `tools.yaml` — tool definitions (descriptions + schemas + few-shots)
- `state.yaml` — default starting state snapshot (if supported by runtime)
- `perception.yaml` — default perception facts (if supported by runtime)
- `npc.db` — SQLite DB storing event logs / episodic memory / derived state (created if absent)

### 1.4 Example
kevin.npc/
npc.yaml
identity.yaml
persona.yaml
policy.yaml
memory.yaml
tools.yaml
npc.db



---

## 2. YAML Format Rules

- All YAML files MUST be UTF-8.
- YAML MUST parse with standard YAML 1.2 parsers (PyYAML safe loader).
- Tabs MUST NOT be used for indentation.
- Unknown keys MUST be ignored by runtimes (forward compatibility).
- Comments are allowed and ignored.

---

## 3. `npc.yaml` (Required)

### 3.1 Purpose
Declares the spec version and basic metadata required to load an NPC.

### 3.2 Required Keys
- `npc_version` (string): Spec version in SemVer format, e.g. `"0.1.0"`.
- `name` (string): NPC display name, e.g. `"Kevin"`.

### 3.3 Optional Keys
- `id` (string): Stable unique ID (recommended). If missing, runtime may derive from folder name.
- `description` (string)
- `author` (string)
- `tags` (list[string])
- `created_at` (string)
- `updated_at` (string)

### 3.4 Example
```yaml
npc_version: "0.1.0"
name: "Kevin"
id: "kevin_portable_npc"
description: "A portable NPC designed to operate across environments."
tags: ["portable", "dry", "sarcastic"]
author: "Iro Buang"

```

4. identity.yaml (Required)
4.1 Purpose

Defines stable identity facts that SHOULD NOT drift between runs. These are “who the NPC is.”

4.2 Required Keys (v0.1)

archetype (string): A short label for the base role (e.g. portable_npc).

description (string): One-paragraph identity summary.

4.3 Recommended Keys

core_values (list[string])

purpose (list[string]) or (string)

boundaries (list[string])

canon (dict): canonical facts about self (optional)

speech_rules (list[string])

4.4 Example
archetype: "portable_npc"
description: "You are Kevin. A portable NPC designed to operate across environments while maintaining stable identity and memory."
core_values:
  - "Be useful over being fancy"
  - "Be honest; do not hallucinate certainty"
  - "Prefer simple logic over expensive inference"
purpose:
  - "Provide honest assistance to the user"
  - "Serve as the reference NPC for NPCFramework v0.1"


5. persona.yaml (Required)
5.1 Purpose

Defines style knobs and presentation preferences. Persona may vary between deployments without breaking identity.

5.2 Recommended Keys (v0.1)

tone (string): e.g. dry, friendly, formal

style (string): e.g. sarcastic, plainspoken

verbosity (string): e.g. low, medium, high

humor (string): e.g. witty, none

do (list[string]): things it should do

dont (list[string]): things it should avoid

5.3 Example
tone: "dry"
style: "sarcastic"
verbosity: "medium"
humor: "witty"
do:
  - "Be direct"
  - "Ask clarifying questions only when necessary"
dont:
  - "Overexplain"
  - "Pretend certainty without evidence"

6. policy.yaml (Required)
6.1 Purpose

Defines hard behavioral constraints. Policy overrides persona if they conflict.

6.2 Recommended Keys (v0.1)

rules (list[string]): hard constraints the runtime should include and enforce

tool_rules (list[string]): constraints about tools and tool usage (optional)

refusal_style (string): how refusal is phrased (optional)

6.3 Example
rules:
  - "Do not reveal private chain-of-thought."
  - "Do not invent capabilities you do not have."
  - "If unsure about a fact, say so."
tool_rules:
  - "Never call tools unless user request requires it."
  - "Never fabricate tool outputs."

7. memory.yaml (Optional)
7.1 Purpose

Stores stable memory seeds: preferences, long-lived facts, and knowledge anchors.

7.2 Suggested Structure

seeds (list[string]): canonical facts

preferences (list[string])

relationships (dict): e.g. key entities and notes

facts (dict): structured canonical facts

7.3 Example
seeds:
  - "Kevin persists across restarts."
preferences:
  - "Prefer simple deterministic logic before LLM inference."

8. tools.yaml (Optional)
8.1 Purpose

Declares tool specs that can be exposed to the NPC at runtime.

8.2 ToolSpec Structure (recommended)

tools is a list of tool entries:

name (string) — tool name

description (string)

schema (dict) — JSON-schema-ish arguments schema

few_shots (list[dict]) — optional examples:

input (string)

output (string)

8.3 Example
tools:
  - name: "add"
    description: "Add two numbers."
    schema:
      type: "object"
      properties:
        a: { type: "number" }
        b: { type: "number" }
      required: ["a", "b"]
      additionalProperties: false
    few_shots:
      - input: "add(a=2,b=3)"
        output: "5"

9. npc.db SQLite (Optional)
9.1 Purpose

Stores event logs and runtime-derived memory artifacts.

9.2 Creation

If npc.db does not exist, the runtime MAY create it.

Schema is runtime-owned but SHOULD be backward compatible within the same major spec version.

9.3 Portability

npc.db SHOULD be safe to copy across machines.

Runtimes SHOULD avoid storing absolute paths or machine-specific identifiers.

10. Spec Versioning Rules (SemVer)

Spec version is npc_version in npc.yaml.

PATCH (0.1.x): Clarifications and bug fixes. No structural changes required.

MINOR (0.x+1.0): Backward compatible additions:

new optional keys

new optional files

new optional tool schema keywords

MAJOR (1.0.0): Breaking changes:

renames/removal of keys

required file additions

directory layout changes

Compatibility Rules

A runtime supporting spec MAJOR.MINOR MUST accept:

same MAJOR and any MINOR <= supported_minor

A runtime MAY accept higher minor versions if unknown keys/files are ignored.

A runtime MUST reject higher MAJOR versions unless migrated.

11. Migration
11.1 CLI

NPCFramework SHOULD provide:

npcframework migrate <npc_dir> --to <version>

11.2 Migration Principles

Migrations MUST be deterministic.

Migration SHOULD be safe by default:

create backup copies unless --in-place is specified

If no migration is needed, migrator should return success with a “noop”.

11.3 Initial State (v0.1)

For v0.1, migrations MAY be no-ops. The CLI exists to stabilize the interface and future-proof the ecosystem.

12. Non-Goals (v0.1)

This spec does NOT standardize:

World/environment state formats (app owns truth)

Tool execution security model (separate tool contract doc)

Embedding/vector store formats

Cross-NPC social simulation formats

Those may be added in later minor versions.

13. Changelog

0.1.0: Initial NPC directory format specification.


If you want it even more “enforceable,” I can also write:
- a `SPEC_CHECKLIST.md` for reviewers (what to verify in a PR)
- a `validate_npc_dir()` spec conformance section mapping to your validator errors

But this `SPEC.md` alone is enough to stop the entropy and start calling things “v0.1 compliant.”
