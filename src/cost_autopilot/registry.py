"""Model registry with real pricing, and the tier->model routing map (YAML)."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import ModelConfig, QualityTier, Tier

# Pricing snapshot (USD / 1M tokens) — update via data/models.yaml without code changes.
DEFAULT_MODELS: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider="openai", model_id="gpt-4o",
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
        avg_latency_ms=1200, quality=QualityTier.high),
    "gpt-4o-mini": ModelConfig(
        provider="openai", model_id="gpt-4o-mini",
        cost_per_1m_input=0.15, cost_per_1m_output=0.60,
        avg_latency_ms=800, quality=QualityTier.medium),
    "claude-sonnet-4-6": ModelConfig(
        provider="anthropic", model_id="claude-sonnet-4-6",
        cost_per_1m_input=3.00, cost_per_1m_output=15.00,
        avg_latency_ms=1300, quality=QualityTier.high),
    "claude-haiku-4-5": ModelConfig(
        provider="anthropic", model_id="claude-haiku-4-5-20251001",
        cost_per_1m_input=1.00, cost_per_1m_output=5.00,
        avg_latency_ms=600, quality=QualityTier.medium),
    "llama-local": ModelConfig(
        provider="ollama", model_id="qwen3:1.7b-q4_K_M",
        cost_per_1m_input=0.0, cost_per_1m_output=0.0,
        avg_latency_ms=2500, quality=QualityTier.low),
}

DEFAULT_ROUTING: dict[Tier, str] = {
    Tier.simple: "llama-local",
    Tier.moderate: "gpt-4o-mini",
    Tier.complex: "gpt-4o",
}

# The "what if everything went to the best model" baseline for savings math.
BASELINE_MODEL = "gpt-4o"


class Registry:
    def __init__(
        self,
        models: dict[str, ModelConfig] | None = None,
        routing: dict[Tier, str] | None = None,
    ):
        self.models = dict(models or DEFAULT_MODELS)
        self.routing = dict(routing or DEFAULT_ROUTING)

    def model_for_tier(self, tier: Tier) -> ModelConfig:
        return self.models[self.routing[tier]]

    def reference_model(self) -> ModelConfig:
        return self.models[BASELINE_MODEL]

    def escalation_for(self, tier: Tier) -> ModelConfig | None:
        """Next tier up, or None if already at the top."""
        if tier == Tier.complex:
            return None
        return self.model_for_tier(Tier(tier + 1))

    def update_routing(self, mapping: dict[int, str]) -> None:
        for tier_value, key in mapping.items():
            if key not in self.models:
                raise KeyError(f"unknown model key: {key}")
            self.routing[Tier(int(tier_value))] = key

    @classmethod
    def from_yaml(cls, path: Path) -> "Registry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        models = {
            key: ModelConfig.model_validate(cfg) for key, cfg in data["models"].items()
        }
        routing = {Tier(int(t)): key for t, key in data["routing"].items()}
        return cls(models, routing)
