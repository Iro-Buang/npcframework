from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import re

from llama_cpp import Llama

Message = Dict[str, str]

# Compile once, not inside a function
_SPEAKER_RE = re.compile(r"^\s*(Kevin\s*>|KEVIN\s*>|Kevin>|KEVIN>)\s*")


@dataclass
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    n_gpu_layers: int = 0          # 0 = CPU
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256

    # Stop tokens are part of config so you can tweak per-model
    stop: Optional[List[str]] = None


def strip_speaker_prefix(text: Optional[str]) -> str:
    """Remove 'Kevin>' style prefixes without nuking whitespace beyond it."""
    if not text:
        return ""
    return _SPEAKER_RE.sub("", text, count=1)


def messages_to_prompt(messages: List[Message]) -> str:
    """
    Transparent prompt formatting.
    Not perfect for every model, but clear and debuggable.
    """
    sys_blocks: List[str] = []
    convo_blocks: List[str] = []

    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()

        if role == "system":
            if content:
                sys_blocks.append(content)
        elif role == "user":
            convo_blocks.append(f"[USER]\n{content}")
        elif role == "assistant":
            convo_blocks.append(f"[ASSISTANT]\n{content}")
        else:
            convo_blocks.append(f"[USER]\n{content}")

    # Stronger anti-echo rules (Gemma likes explicit constraints)
    sys_text = "\n\n".join(sys_blocks).strip()
    if sys_text:
        sys_text += "\n\n"
    sys_text += (
        "RULES:\n"
        "- Reply ONLY as the assistant.\n"
        "- Do NOT repeat or quote the user’s message.\n"
        "- Do NOT include speaker labels like 'Kevin>' or '[ASSISTANT]'.\n"
        "- If you don’t know an entity, say you don’t know and ask a question.\n"
    )

    prompt_parts: List[str] = []
    prompt_parts.append(f"<<SYS>>\n{sys_text}\n<</SYS>>")
    prompt_parts.extend(convo_blocks)
    prompt_parts.append("[ASSISTANT]\n")  # force completion as assistant

    return "\n\n".join(prompt_parts)


class LlamaCppEngine:
    def __init__(self, cfg: LlamaCppConfig) -> None:
        self.cfg = cfg
        self.llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )

    def _stop_tokens(self) -> List[str]:
        # Default stop tokens if none provided
        return self.cfg.stop or ["[USER]", "<<SYS>>", "<</SYS>>", "[ASSISTANT]"]

    def chat_stream(self, messages: List[Message]) -> Iterable[str]:
        prompt = messages_to_prompt(messages)
        stop = self._stop_tokens()

        stream = self.llm(
            prompt=prompt,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            stop=stop,
            stream=True,
        )

        first = True
        for chunk in stream:
            # llama-cpp-python sometimes yields chunks with no text
            if not chunk or "choices" not in chunk:
                continue

            piece = chunk["choices"][0].get("text")
            if piece is None or piece == "":
                continue

            # Always strip speaker labels
            piece = strip_speaker_prefix(piece)

            if first:
                # Remove ONLY leading newlines (not spaces) for pretty terminal output
                piece = piece.lstrip("\r\n")
                first = False

            if piece:
                yield piece

    def chat(self, messages: List[Message]) -> str:
        return "".join(self.chat_stream(messages)).strip()
