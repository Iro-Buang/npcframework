Alright, here’s the boringly correct, future-you-proof version.
Print this, tattoo it on the repo, and never reopen this rabbit hole again.

📜 NPCFramework Tool Result Contract (v1.0)

Purpose:
Prevent hallucinations, double tool calls, creative math, and “06:04 PM forever” bugs.
Tools return truth. Models decorate truth. Models do not invent truth.

1. Tool Return Shape (MANDATORY)

Every tool MUST return a dictionary with this shape:

{
  "answer": <scalar>,
  "meta": { ... }  # optional
}

Allowed answer types

str

int

float

bool

❌ No lists
❌ No nested objects
❌ No “maybe” fields
❌ No vibes

If it can’t fit in answer, it belongs in meta.

2. Tool Runtime Envelope (SYSTEM-OWNED)

The runtime wraps tool output into this non-negotiable structure:

{
  "ok": true,
  "tool": "time_now",
  "result": {
    "answer": "06:54 PM",
    "meta": {
      "iso": "2025-12-20T18:54:12+08:00"
    }
  },
  "error": null,
  "latency_ms": 11
}


result.answer is the canonical truth

meta is for debugging, logging, analytics, vibes

latency_ms is for guilt-tripping slow tools

3. Canonical Answer Rule (CRITICAL)

After any successful tool call:

result.answer is ground truth

The model MUST include it verbatim

The model MAY NOT:

reformat numbers

rephrase times

approximate

“explain” the value

improve it (💀)

If the user asked:

“what time is it”

And the tool says:

"answer": "06:54 PM"


Then the final output must include:

06:54 PM


Anything else is a bug.

4. Model Behavior Rules (Post-Tool)

Injected after tool result:

Tool results are authoritative

Never invent values already provided

If answer exists, use it

Style and tone are allowed around the answer, not instead of it

Good:

“It’s currently 06:54 PM.”

Bad:

“It looks like it’s around evening, maybe 6-ish.”

Instant jail.

5. Standard Tool Examples
time_now
def time_now(_: dict) -> dict:
    now = datetime.now().astimezone()
    return {
        "answer": now.strftime("%I:%M %p"),
        "meta": {
            "iso": now.isoformat()
        }
    }

add
def add(args: dict) -> dict:
    return {
        "answer": float(args["a"]) + float(args["b"]),
        "meta": {
            "a": args["a"],
            "b": args["b"]
        }
    }

6. What NOT To Do (Learned the Hard Way)

❌ Returning raw scalars
❌ Returning { "time": "06:04 PM" }
❌ Expecting the model to “figure it out”
❌ Regexing the model output
❌ Trusting Gemma/LLaMA “because it worked once”

The model is not a calculator.
The model is not a clock.
The model is a formatter with opinions.

7. Design Philosophy (Why This Exists)

Tools are deterministic truth engines
Models are narrative surfaces

NPCFramework draws the line hard so:

logs are auditable

bugs are reproducible

hallucinations are impossible by construction

future demos don’t embarrass you on stage

8. Versioning

Contract version: v1.0

Breaking changes require:

migration note

extractor update

one sigh from Future You