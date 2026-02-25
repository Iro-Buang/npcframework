from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional

Message = Dict[str, str]


@dataclass(frozen=True)
class VLLMConfig:
    """Config for vLLM OpenAI-compatible server mode."""
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = ""
    api_key: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    timeout_s: float = 60.0
    # NPCFramework expects a streaming-like interface; we can still do non-stream and yield once.
    stream: bool = False


class VLLMEngine:
    """Inference adapter for vLLM when running as an OpenAI-compatible HTTP server.

    Notes:
    - Designed for 'plug and play' server mode.
    - Emits a single chunk if stream=False.
    - If stream=True, vLLM returns SSE; we parse it and yield incremental deltas.
      This is best-effort and avoids external deps.
    """

    def __init__(self, cfg: VLLMConfig) -> None:
        if not cfg.base_url:
            raise ValueError("VLLMConfig.base_url is required")
        if not cfg.model:
            raise ValueError("VLLMConfig.model is required (HF repo id or local HF folder served by vLLM)")
        self.cfg = cfg

    def chat_stream(self, messages: List[Message]) -> Iterable[str]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
            "stream": bool(self.cfg.stream),
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                if not self.cfg.stream:
                    data = resp.read().decode("utf-8", errors="replace")
                    obj = json.loads(data)
                    content = (((obj.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
                    if content:
                        yield content
                    return

                # SSE stream: lines like "data: {...}\n\n"
                buffer = ""
                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            return
                        try:
                            obj = json.loads(data)
                            delta = (((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")) or ""
                            if delta:
                                yield delta
                        except Exception:
                            # ignore malformed chunks
                            continue

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"vLLM HTTPError {e.code}: {body or e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"vLLM connection error: {e}") from e
