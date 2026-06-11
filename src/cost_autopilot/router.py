"""The routing core: classify -> pick model -> call -> log -> (async) verify."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from .classifier import classify
from .providers import ProviderHub
from .registry import Registry
from .schemas import LLMResponse, RequestLog, RouteDecision
from .storage import Store
from .verifier import Verifier


class Router:
    def __init__(
        self,
        registry: Registry | None = None,
        hub: ProviderHub | None = None,
        store: Store | None = None,
        verify_enabled: bool = True,
    ):
        self.registry = registry or Registry()
        self.hub = hub or ProviderHub()
        self.store = store or Store()
        self.verifier = Verifier(self.registry, self.hub)
        self.verify_enabled = verify_enabled

    def decide(self, prompt: str) -> RouteDecision:
        tier, features = classify(prompt)
        cfg = self.registry.model_for_tier(tier)
        return RouteDecision(
            tier=tier, model_id=cfg.model_id,
            reason=f"tier {int(tier)} -> {cfg.model_id} "
                   f"(verbs={features.instruction_verbs}, "
                   f"constraints={features.constraint_count}, "
                   f"fmt={features.output_format_complexity})",
            features=features,
        )

    def complete(self, prompt: str) -> tuple[LLMResponse, RouteDecision, RequestLog]:
        decision = self.decide(prompt)
        cfg = self.registry.models[
            next(k for k, m in self.registry.models.items()
                 if m.model_id == decision.model_id)
        ]
        response = self.hub.send(prompt, cfg)
        request_id = uuid.uuid4().hex[:12]
        baseline_cfg = self.registry.reference_model()
        baseline_cost = (
            response.tokens_in * baseline_cfg.cost_per_1m_input
            + response.tokens_out * baseline_cfg.cost_per_1m_output
        ) / 1_000_000

        escalated = False
        if self.verify_enabled:
            verif, better = self.verifier.verify(
                request_id, prompt, response, decision.tier)
            self.store.log_verification(verif)
            if verif.routing_failure:
                self.store.log_routing_failure(
                    request_id, prompt, wrong_tier=int(decision.tier))
            if better is not None:
                response = better
                escalated = True

        log = RequestLog(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            tier=int(decision.tier),
            routed_model=response.model_id,
            cost=response.cost(self._cfg_for(response.model_id)),
            latency_ms=response.latency_ms,
            escalated=escalated,
            baseline_cost=baseline_cost,
        )
        self.store.log_request(log)
        return response, decision, log

    def _cfg_for(self, model_id: str):
        for cfg in self.registry.models.values():
            if cfg.model_id == model_id:
                return cfg
        return self.registry.reference_model()
