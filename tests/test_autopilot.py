"""Classifier, routing, verification loop, savings math, API."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from cost_autopilot.api import create_app
from cost_autopilot.classifier import classify, evaluate
from cost_autopilot.providers import DeterministicProvider, ProviderHub
from cost_autopilot.registry import Registry
from cost_autopilot.router import Router
from cost_autopilot.schemas import Tier
from cost_autopilot.storage import Store
from cost_autopilot.verifier import agreement_score

ROOT = Path(__file__).resolve().parents[1]


def test_classifier_tiers_on_canonical_prompts():
    tier, _ = classify("Extrais l'adresse email de ce texte : contact@x.fr")
    assert tier == Tier.simple
    tier, _ = classify(
        "Analyse les avantages et inconvénients de deux architectures, compare les "
        "coûts et argumente une recommandation, réponds en JSON structuré.")
    assert tier == Tier.complex


def test_classifier_accuracy_above_80_pct_on_labeled_set():
    labeled = json.loads((ROOT / "data" / "labeled_prompts.json").read_text(encoding="utf-8"))
    report = evaluate(labeled)
    assert report["n"] >= 20
    assert report["accuracy"] >= 0.8, report


def test_registry_routing_and_escalation():
    reg = Registry()
    assert reg.model_for_tier(Tier.simple).cost_per_1m_input == 0.0
    esc = reg.escalation_for(Tier.simple)
    assert esc is not None and esc.quality > reg.model_for_tier(Tier.simple).quality
    assert reg.escalation_for(Tier.complex) is None


def test_routing_map_hot_update():
    reg = Registry()
    reg.update_routing({1: "claude-haiku-4-5"})
    assert reg.model_for_tier(Tier.simple).model_id.startswith("claude-haiku")


def test_agreement_score_behaviour():
    assert agreement_score("le chat dort sur le tapis", "le chat dort sur le tapis") == 1.0
    assert agreement_score("réponse complètement différente", "autre chose sans rapport") < 0.5


def test_router_logs_and_baseline_cost():
    router = Router(store=Store(":memory:"))
    response, decision, log = router.complete("Extrais la date de ce texte : 12 mai 2026.")
    assert response.text
    assert log.tier == int(decision.tier)
    assert log.baseline_cost >= log.cost  # cheap routing never costs more than baseline
    stats = router.store.stats()
    assert stats["requests"] == 1


def test_savings_accumulate_over_load():
    router = Router(store=Store(":memory:"))
    prompts = [
        "Extrais le numéro de facture : INV-001",
        "Traduis : bonjour le monde",
        "Résume ce texte en deux phrases : la réunion a duré trois heures et a produit cinq décisions.",
        "Analyse et compare deux stratégies de cache, argumente une recommandation en JSON.",
    ] * 5
    for p in prompts:
        router.complete(p)
    stats = router.store.stats()
    assert stats["requests"] == 20
    assert stats["savings_pct"] > 0
    assert len(stats["routing_distribution"]) >= 2


def test_verification_failure_feeds_flywheel():
    router = Router(store=Store(":memory:"), verify_enabled=True)
    # Force disagreement: cheap deterministic model keeps only 6 words, so a long
    # complex prompt routed to tier 1 diverges from the reference output.
    router.verifier.threshold = 0.99
    router.complete("Extrais le point clé : " + " ".join(f"mot{i}" for i in range(40)))
    failures = router.store.failure_examples()
    assert len(failures) >= 1


def test_api_endpoints():
    app = create_app(Router(Registry(), ProviderHub(), Store(":memory:")))
    client = TestClient(app)
    r = client.post("/v1/completions", json={"prompt": "Traduis : bonjour"})
    assert r.status_code == 200
    body = r.json()
    assert "model" in body and "routing" in body
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/stats").json()["requests"] == 1
    r = client.put("/v1/routing-config", json={"routing": {1: "gpt-4o-mini"}})
    assert r.status_code == 200
    assert r.json()["routing"]["1"] == "gpt-4o-mini"


def test_deterministic_provider_quality_gradient():
    from cost_autopilot.registry import DEFAULT_MODELS
    prov = DeterministicProvider()
    prompt = " ".join(f"mot{i}" for i in range(50))
    cheap = prov.send(prompt, DEFAULT_MODELS["llama-local"])
    rich = prov.send(prompt, DEFAULT_MODELS["gpt-4o"])
    assert len(rich.text) > len(cheap.text)
