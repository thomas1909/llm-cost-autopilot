"""FastAPI service: the caller never picks the model — the router does."""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from .providers import ProviderHub
from .registry import Registry
from .router import Router
from .storage import Store


class CompletionRequest(BaseModel):
    prompt: str


class RoutingConfigUpdate(BaseModel):
    routing: dict[int, str]  # tier -> registry key


def create_app(router: Router | None = None) -> FastAPI:
    app = FastAPI(title="Cost Autopilot", version="0.1.0")
    if router is None:
        hub = ProviderHub(
            openai_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY", ""),
            ollama_url=os.getenv("OLLAMA_BASE_URL", ""),
        )
        router = Router(Registry(), hub, Store(os.getenv("DB_PATH", "data/autopilot.db")))
    app.state.router = router

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        response, decision, log = app.state.router.complete(req.prompt)
        return {
            "text": response.text,
            "model": response.model_id,
            "routing": {
                "tier": int(decision.tier),
                "reason": decision.reason,
                "escalated": log.escalated,
            },
            "cost_usd": log.cost,
            "latency_ms": log.latency_ms,
        }

    @app.get("/v1/models")
    def models():
        r: Router = app.state.router
        return {
            key: {
                "provider": cfg.provider, "model_id": cfg.model_id,
                "cost_per_1m_input": cfg.cost_per_1m_input,
                "cost_per_1m_output": cfg.cost_per_1m_output,
                "quality": int(cfg.quality),
            }
            for key, cfg in r.registry.models.items()
        }

    @app.get("/v1/stats")
    def stats():
        return app.state.router.store.stats()

    @app.put("/v1/routing-config")
    def update_routing(update: RoutingConfigUpdate):
        app.state.router.registry.update_routing(update.routing)
        return {"routing": {int(t): k for t, k in app.state.router.registry.routing.items()}}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app() if os.getenv("AUTOPILOT_AUTOSTART") else None
