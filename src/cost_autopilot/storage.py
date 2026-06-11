"""SQLite audit trail: every routed request, every verification, every escalation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .schemas import RequestLog, VerificationResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    tier INTEGER NOT NULL,
    routed_model TEXT NOT NULL,
    cost REAL NOT NULL,
    latency_ms REAL NOT NULL,
    escalated INTEGER NOT NULL DEFAULT 0,
    quality_score REAL,
    baseline_cost REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS verifications (
    request_id TEXT PRIMARY KEY,
    cheap_model TEXT NOT NULL,
    reference_model TEXT NOT NULL,
    agreement REAL NOT NULL,
    routing_failure INTEGER NOT NULL,
    escalated INTEGER NOT NULL,
    cost_delta REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS routing_failures (
    request_id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    wrong_tier INTEGER NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    def log_request(self, log: RequestLog) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?,?,?,?)",
            (log.request_id, log.timestamp, log.prompt_hash, log.tier,
             log.routed_model, log.cost, log.latency_ms, int(log.escalated),
             log.quality_score, log.baseline_cost),
        )
        self._conn.commit()

    def log_verification(self, v: VerificationResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO verifications VALUES (?,?,?,?,?,?,?)",
            (v.request_id, v.cheap_model, v.reference_model, v.agreement,
             int(v.routing_failure), int(v.escalated), v.cost_delta),
        )
        self._conn.commit()

    def log_routing_failure(self, request_id: str, prompt: str, wrong_tier: int) -> None:
        """Failures become future training examples for the classifier (the flywheel)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO routing_failures VALUES (?,?,?)",
            (request_id, prompt, wrong_tier),
        )
        self._conn.commit()

    def stats(self) -> dict:
        cur = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost),0), COALESCE(SUM(baseline_cost),0), "
            "COALESCE(SUM(escalated),0) FROM requests"
        )
        n, cost, baseline, escalated = cur.fetchone()
        dist = dict(self._conn.execute(
            "SELECT routed_model, COUNT(*) FROM requests GROUP BY routed_model"
        ).fetchall())
        savings = (baseline - cost) / baseline if baseline else 0.0
        return {
            "requests": n,
            "total_cost_usd": round(cost, 6),
            "baseline_cost_usd": round(baseline, 6),
            "savings_pct": round(savings * 100, 2),
            "escalations": escalated,
            "routing_distribution": dist,
        }

    def failure_examples(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT prompt, wrong_tier FROM routing_failures"
        ).fetchall()
        return [{"prompt": p, "tier": t} for p, t in rows]
