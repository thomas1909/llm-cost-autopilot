"""Async quality verification loop.

After the cheap answer is returned, the same prompt goes to the reference model;
agreement below the threshold = routing failure -> escalate and feed the failure
back as a classifier training example.
"""

from __future__ import annotations

import re

from .providers import ProviderHub
from .registry import Registry
from .schemas import LLMResponse, Tier, VerificationResult


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[\wà-ÿ]+", text.lower()) if len(t) > 2}


def agreement_score(a: str, b: str) -> float:
    """Jaccard similarity of content tokens — provider-agnostic, deterministic."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class Verifier:
    def __init__(self, registry: Registry, hub: ProviderHub, threshold: float = 0.35):
        self.registry = registry
        self.hub = hub
        self.threshold = threshold

    def verify(
        self,
        request_id: str,
        prompt: str,
        cheap_response: LLMResponse,
        tier: Tier,
    ) -> tuple[VerificationResult, LLMResponse | None]:
        """Returns the verification result and the escalated response, if any."""
        reference_cfg = self.registry.reference_model()
        if cheap_response.model_id == reference_cfg.model_id:
            result = VerificationResult(
                request_id=request_id, cheap_model=cheap_response.model_id,
                reference_model=reference_cfg.model_id, agreement=1.0,
                routing_failure=False, escalated=False, cost_delta=0.0,
            )
            return result, None

        reference = self.hub.send(prompt, reference_cfg)
        agreement = agreement_score(cheap_response.text, reference.text)
        failure = agreement < self.threshold
        escalated_resp: LLMResponse | None = None
        cost_delta = 0.0
        if failure:
            escalation_cfg = self.registry.escalation_for(tier) or reference_cfg
            escalated_resp = self.hub.send(prompt, escalation_cfg)
            cheap_cfg = self.registry.models.get(cheap_response.model_id)
            cheap_cost = cheap_response.cost(cheap_cfg) if cheap_cfg else 0.0
            cost_delta = escalated_resp.cost(escalation_cfg) - cheap_cost

        result = VerificationResult(
            request_id=request_id,
            cheap_model=cheap_response.model_id,
            reference_model=reference_cfg.model_id,
            agreement=round(agreement, 4),
            routing_failure=failure,
            escalated=escalated_resp is not None,
            cost_delta=round(cost_delta, 8),
        )
        return result, escalated_resp
