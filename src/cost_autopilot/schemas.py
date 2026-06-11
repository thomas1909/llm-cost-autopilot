"""Typed contracts: model registry entries, routing decisions, audit rows."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field


class Tier(IntEnum):
    simple = 1      # reformatting, extraction, basic Q&A over provided context
    moderate = 2    # summarization, classification, structured analysis
    complex = 3     # multi-step reasoning, creative generation, nuanced judgment


class QualityTier(IntEnum):
    low = 1
    medium = 2
    high = 3


class ModelConfig(BaseModel):
    provider: str               # openai | anthropic | ollama | deterministic
    model_id: str
    cost_per_1m_input: float    # USD
    cost_per_1m_output: float   # USD
    avg_latency_ms: float
    quality: QualityTier


class LLMResponse(BaseModel):
    text: str
    model_id: str
    provider: str
    tokens_in: int
    tokens_out: int
    latency_ms: float

    def cost(self, cfg: ModelConfig) -> float:
        return (
            self.tokens_in * cfg.cost_per_1m_input
            + self.tokens_out * cfg.cost_per_1m_output
        ) / 1_000_000


class Features(BaseModel):
    """Features extracted from a prompt for complexity scoring."""

    token_count: int
    instruction_verbs: int
    constraint_count: int
    has_context: bool
    output_format_complexity: int  # 0 plain, 1 list/table, 2 structured/json/code
    question_marks: int


class RouteDecision(BaseModel):
    tier: Tier
    model_id: str
    reason: str
    features: Features


class VerificationResult(BaseModel):
    request_id: str
    cheap_model: str
    reference_model: str
    agreement: float = Field(ge=0.0, le=1.0)
    routing_failure: bool
    escalated: bool
    cost_delta: float


class RequestLog(BaseModel):
    request_id: str
    timestamp: str
    prompt_hash: str
    tier: int
    routed_model: str
    cost: float
    latency_ms: float
    escalated: bool = False
    quality_score: float | None = None
    baseline_cost: float = 0.0   # what GPT-4o-class routing would have cost
