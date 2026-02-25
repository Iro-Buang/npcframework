from __future__ import annotations
from typing import Dict, List
from typing import Iterable, List, Dict

Message = Dict[str, str]

class MockEngine:
    """
    Deterministic mock inference engine.

    - Implements BOTH chat() and chat_stream()
    - chat() is implemented via chat_stream() to mirror real engines
    """

    def chat_stream(self, messages: List[Message]) -> Iterable[str]:
        # Super dumb deterministic reply, on purpose
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        reply = f"(mock) You said: {last_user or 'nothing'}"

        # Stream it as a single chunk
        yield reply

    def chat(self, messages: List[Message]) -> str:
        return "".join(self.chat_stream(messages))