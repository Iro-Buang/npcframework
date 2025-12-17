from __future__ import annotations
from typing import Dict, List

Message = Dict[str, str]

class MockEngine:
    def __init__(self, name: str = "Kevin"):
        self.name = name

    def chat(self, messages: List[Message]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if last_user.lower() in ("hi", "hello", "hey"):
            return f"{self.name}> Sup. Try saying something that requires more than two brain cells."
        if len(last_user) < 4:
            return f"{self.name}> That’s not a thought. That’s a sneeze."
        return f"{self.name}> Noted. Now tell me what you *actually* want to do."
