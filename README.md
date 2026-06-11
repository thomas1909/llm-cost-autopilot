# Cost Autopilot — routage LLM piloté par les coûts

> **Projet 2 du guide BASWE.** Une couche de routage qui analyse la complexité de
> chaque requête, l'envoie au **modèle le moins cher capable de la traiter**, puis
> vérifie en asynchrone que la décision était correcte — avec escalade automatique
> et boucle d'apprentissage sur les échecs de routage.

## Architecture

```
requête ─> classifier (features → tier 1/2/3) ─> routing map (YAML hot-reload)
              │                                        │
              │                                  provider hub (OpenAI/Anthropic/Ollama
              │                                        │        ou déterministe offline)
              │                                  réponse + coût + latence
              │                                        │
              └── flywheel <── routing_failures <── verifier (vs modèle de référence)
                  (les échecs deviennent                  │
                   des exemples étiquetés)          escalade auto si désaccord
```

- **Registre de modèles** (`registry.py`) : pricing réel par 1M tokens (GPT-4o,
  GPT-4o-mini, Claude Sonnet/Haiku, modèle local Ollama gratuit), tier de qualité,
  latence moyenne. Routing map modifiable à chaud via `PUT /v1/routing-config`.
- **Classifieur de complexité** (`classifier.py`) : extraction de features
  (verbes d'instruction, contraintes, format de sortie, contexte fourni, longueur)
  + scoring pondéré transparent. **≥ 80 % d'accuracy** sur le dataset étiqueté
  `data/labeled_prompts.json` (vérifié par les tests). Les features sont prêtes
  pour une régression logistique sklearn si besoin.
- **Vérification async** (`verifier.py`) : la même requête part vers le modèle de
  référence ; accord mesuré par similarité Jaccard. Désaccord → échec de routage
  logué + **escalade automatique** vers le tier supérieur + l'exemple rejoint la
  table `routing_failures` pour ré-entraîner le classifieur (le flywheel).
- **Audit SQLite** (`storage.py`) : chaque requête avec coût réel **et coût
  baseline** ("et si tout passait par GPT-4o ?") → `GET /v1/stats` expose le
  pourcentage d'économies, la métrique reine du projet.

## Offline-first

Sans clé API, tous les modèles résolvent vers un backend déterministe dont la
qualité **dégrade réellement avec le tier** (les modèles bas de gamme tronquent la
réponse), ce qui permet d'exercer le chemin de désaccord/escalade sans réseau.

## Démarrage

```bash
uv sync --extra dev --link-mode=copy
uv run --no-sync pytest -v          # 10 tests offline

# Lancer l'API
$env:AUTOPILOT_AUTOSTART="1"; uv run --no-sync uvicorn cost_autopilot.api:app --port 8200
```

```bash
curl -X POST localhost:8200/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt": "Extrais la date : réunion le 12 mai"}'
curl localhost:8200/v1/stats        # → savings_pct, routing_distribution, escalations
```

## Variables d'environnement

| Variable | Effet |
|---|---|
| `OPENAI_API_KEY` | active le provider OpenAI réel |
| `ANTHROPIC_API_KEY` | active le provider Anthropic réel |
| `OLLAMA_BASE_URL` | active Ollama (ex: `http://localhost:11434`) |
| `DB_PATH` | chemin SQLite (défaut `data/autopilot.db`) |

Sans aucune variable : 100 % offline, déterministe.
