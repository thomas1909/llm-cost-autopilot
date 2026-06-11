"""Unified send_request() across providers, with an offline deterministic backend.

Real providers (OpenAI / Anthropic / Ollama) are called through httpx only when
credentials / endpoints are configured; otherwise every model resolves to the
deterministic backend so the whole system runs offline.
"""

from __future__ import annotations

import hashlib
import time
from typing import Protocol

import httpx

from .schemas import LLMResponse, ModelConfig


class Provider(Protocol):
    def send(self, prompt: str, cfg: ModelConfig) -> LLMResponse: ...


class DeterministicProvider:
    """Offline stand-in: answer quality degrades with cheaper model tiers,
    which lets the verification loop exercise real disagreement paths."""

    def send(self, prompt: str, cfg: ModelConfig) -> LLMResponse:
        start = time.perf_counter()
        words = prompt.split()
        # Higher-quality tiers "see" more of the prompt in their answer.
        keep = {1: 6, 2: 12, 3: 24}[int(cfg.quality)]
        digest = hashlib.sha1(prompt.encode()).hexdigest()[:6]
        text = (
            f"[{cfg.model_id}] réponse({digest}): " + " ".join(words[:keep])
        )
        latency = (time.perf_counter() - start) * 1000
        return LLMResponse(
            text=text, model_id=cfg.model_id, provider=cfg.provider,
            tokens_in=len(words), tokens_out=len(text.split()),
            latency_ms=latency,
        )


class OpenAICompatibleProvider:
    """OpenAI-style chat completions endpoint (works for OpenAI and Ollama)."""

    def __init__(self, base_url: str, api_key: str = ""):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=120)

    def send(self, prompt: str, cfg: ModelConfig) -> LLMResponse:
        start = time.perf_counter()
        resp = self._client.post("/chat/completions", json={
            "model": cfg.model_id,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model_id=cfg.model_id, provider=cfg.provider,
            tokens_in=usage.get("prompt_tokens", len(prompt.split())),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class AnthropicProvider:
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        self._client = httpx.Client(
            base_url=base_url, timeout=120,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )

    def send(self, prompt: str, cfg: ModelConfig) -> LLMResponse:
        start = time.perf_counter()
        resp = self._client.post("/v1/messages", json={
            "model": cfg.model_id, "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return LLMResponse(
            text="".join(b.get("text", "") for b in data.get("content", [])),
            model_id=cfg.model_id, provider=cfg.provider,
            tokens_in=usage.get("input_tokens", len(prompt.split())),
            tokens_out=usage.get("output_tokens", 0),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class ProviderHub:
    """Resolves a ModelConfig to a concrete provider; deterministic by default."""

    def __init__(self, openai_key: str = "", anthropic_key: str = "",
                 ollama_url: str = ""):
        self._fallback = DeterministicProvider()
        self._providers: dict[str, Provider] = {}
        if openai_key:
            self._providers["openai"] = OpenAICompatibleProvider(
                "https://api.openai.com/v1", openai_key)
        if anthropic_key:
            self._providers["anthropic"] = AnthropicProvider(anthropic_key)
        if ollama_url:
            self._providers["ollama"] = OpenAICompatibleProvider(f"{ollama_url}/v1")

    def send(self, prompt: str, cfg: ModelConfig) -> LLMResponse:
        provider = self._providers.get(cfg.provider, self._fallback)
        return provider.send(prompt, cfg)
