# CLAUDE.md — Cost Autopilot (BASWE Project 2)

## Goal
Cost-aware routing layer: complexity classifier -> cheapest capable model ->
async verification vs reference model -> auto-escalation -> failures feed back
as classifier training data. SQLite audit with baseline-cost savings math.

## Stack
Python 3.11 · `uv` · Pydantic v2 · FastAPI · httpx · PyYAML · SQLite · pytest · ruff.

## Modules (`src/cost_autopilot/`)
- **schemas.py** — Tier(1/2/3), QualityTier, ModelConfig (pricing /1M tokens),
  LLMResponse (.cost(cfg)), Features, RouteDecision, VerificationResult, RequestLog.
- **registry.py** — DEFAULT_MODELS (gpt-4o, gpt-4o-mini, claude sonnet/haiku,
  llama-local at $0), DEFAULT_ROUTING tier->key, BASELINE_MODEL="gpt-4o",
  escalation_for(), update_routing() (hot), from_yaml().
- **classifier.py** — extract_features + weighted scoring -> Tier; evaluate()
  reports accuracy/confusion on data/labeled_prompts.json (≥80% enforced by tests).
- **providers.py** — DeterministicProvider (quality gradient: tier keeps 6/12/24
  words — exercises disagreement offline), OpenAICompatibleProvider (OpenAI+Ollama),
  AnthropicProvider, ProviderHub (resolves by env keys, falls back deterministic).
- **verifier.py** — Jaccard agreement vs reference model; failure -> escalate +
  log routing_failure. Threshold 0.35.
- **router.py** — decide() / complete(): classify, call, verify, log (cost +
  baseline_cost for savings).
- **storage.py** — SQLite: requests / verifications / routing_failures; stats()
  -> savings_pct, routing_distribution.
- **api.py** — POST /v1/completions, GET /v1/models, GET /v1/stats,
  PUT /v1/routing-config, /health. create_app(router) injectable for tests.

## Commands
```bash
uv sync --extra dev --link-mode=copy
uv run --no-sync pytest -v
uv run --no-sync ruff check .
$env:AUTOPILOT_AUTOSTART="1"; uv run --no-sync uvicorn cost_autopilot.api:app --port 8200
```

## Hard rules
- Tests NEVER hit the network (ProviderHub with no keys = deterministic).
- Classifier accuracy ≥ 0.8 on the labeled set is a test invariant.
- baseline_cost >= cost for every logged request (cheap routing never costs more).
- ruff + pytest green before stopping.
