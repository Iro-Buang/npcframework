# NPCFramework Tool Contract (v0.1)

This document defines the tool-calling interface between:
- The NPC runtime (prompt compiler + orchestrator)
- The application (tool implementations)

## 1. Tool name
- Type: `string`
- Pattern: `^[a-zA-Z][a-zA-Z0-9_]{0,63}$`
- Examples: `add`, `search_web`, `db_insert`, `set_light`
- Rules:
  - Must be unique within a turn.
  - Case-sensitive (recommend: lowercase snake_case).
  - No spaces, no hyphens.

## 2. Tool schema (accepted subset)
NPCFramework accepts a JSON-Schema-ish dict (subset).

Required fields:
- `type`: must be `"object"`
- `properties`: dict of `{arg_name: spec}`
Optional fields:
- `required`: list of required argument names
- `additionalProperties`: bool (default false recommended)
- `description`: string (tool and/or property)

Allowed property types:
- `"string"`, `"number"`, `"integer"`, `"boolean"`, `"object"`, `"array"`
Allowed validation keywords (best-effort):
- `enum`, `minLength`, `maxLength`, `minimum`, `maximum`

Anything outside this subset may be ignored.

Example:
```json
{
  "type": "object",
  "properties": {
    "a": {"type": "number"},
    "b": {"type": "number"}
  },
  "required": ["a","b"],
  "additionalProperties": false
}
