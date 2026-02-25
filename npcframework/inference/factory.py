from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from npcframework.npcframework_types import EngineConfig, InferenceEngine
from npcframework.inference.mock import MockEngine


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root (expected mapping): {p}")
    return data


def engine_config_from_yaml(path: str | Path) -> EngineConfig:
    """Load canonical inference config from YAML.

    This is the single source of truth for backend + model selection.
    """
    cfg = _read_yaml(path)

    backend = (cfg.get("backend") or "mock").strip()

    # Common
    model = cfg.get("model") or {}
    if not isinstance(model, dict):
        model = {}

    # Backend-specific
    if backend == "llamacpp":
        llama = cfg.get("llamacpp") or {}
        if not isinstance(llama, dict):
            llama = {}
        return EngineConfig(
            backend="llamacpp",
            model_path=str(model.get("model_path") or ""),
            n_ctx=int(llama.get("n_ctx", 8192)),
            n_threads=llama.get("n_threads", None),
            n_gpu_layers=int(llama.get("n_gpu_layers", 0)),
            temperature=float(llama.get("temperature", 0.7)),
            top_p=float(llama.get("top_p", 0.9)),
            max_tokens=int(llama.get("max_tokens", 256)),
            stop=llama.get("stop", None),
            # vllm fields ignored
        )

    if backend == "vllm":
        v = cfg.get("vllm") or {}
        if not isinstance(v, dict):
            v = {}
        # model key is vllm.model (HF id/folder served by vLLM), NOT gguf
        return EngineConfig(
            backend="vllm",
            model_path=str(model.get("model_path") or ""),
            temperature=float(v.get("temperature", 0.7)),
            top_p=float(v.get("top_p", 0.9)),
            max_tokens=int(v.get("max_tokens", 256)),
            vllm_mode=str(v.get("mode", "server")),
            vllm_base_url=str(v.get("base_url", "http://127.0.0.1:8000/v1")),
            vllm_model=str(v.get("model") or ""),
            vllm_api_key=str(v.get("api_key") or ""),
            vllm_timeout_s=float(v.get("timeout_s", 60.0)),
            vllm_stream=bool(v.get("stream", False)),
        )

    if backend == "mock":
        return EngineConfig(backend="mock")

    raise ValueError(f"Unknown backend in YAML: {backend}")


def load_engine_from_yaml(path: str | Path):
    """Convenience: returns EngineConfig, leaving Engine construction to npcframework.api.Engine."""
    return engine_config_from_yaml(path)
