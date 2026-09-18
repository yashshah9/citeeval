"""Docker integration for citeeval."""

from __future__ import annotations

import os
import time

import httpx
import pytest

BASE = os.environ.get("CITEEVAL_BASE_URL", "http://127.0.0.1:8091")
AUTH = {"Authorization": "Bearer dev-key"}

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_INTEGRATION") != "1",
    reason="Set RUN_DOCKER_INTEGRATION=1 against a live compose stack",
)


def _wait_healthy(timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE}/health", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"citeeval not healthy at {BASE}")


def test_docker_ingest_ask_eval_traces() -> None:
    _wait_healthy()
    httpx.post(
        f"{BASE}/v1/admin/reset",
        headers={"Authorization": "Bearer admin-key"},
        timeout=5.0,
    )
    ing = httpx.post(
        f"{BASE}/v1/ingest",
        headers=AUTH,
        json={
            "source": "policy.md",
            "text": "Refund policy allows returns within 30 days of purchase.",
        },
        timeout=10.0,
    )
    assert ing.status_code == 200
    ask = httpx.post(
        f"{BASE}/v1/ask",
        headers=AUTH,
        json={"question": "refund policy days"},
        timeout=10.0,
    )
    body = ask.json()
    assert body["status"] == "ok"
    assert body["citations"][0]["source"] == "policy.md"
    assert body["trace"]["retrieval_mode"] == "hybrid_hash"
    assert body["trace"]["estimated_cost_usd"] > 0

    traces = httpx.get(f"{BASE}/v1/traces", headers=AUTH, timeout=5.0).json()
    assert traces["count"] >= 1

    ev = httpx.post(
        f"{BASE}/v1/eval",
        headers=AUTH,
        json={
            "cases": [
                {
                    "question": "refund policy",
                    "must_cite_source": "policy.md",
                    "must_include": "30 days",
                }
            ]
        },
        timeout=10.0,
    )
    assert ev.json()["passed"] == 1

    health = httpx.get(f"{BASE}/health", timeout=5.0).json()
    assert health["queue"] == "redis"
    assert health["audit"] == "postgres"
    assert health["corpus"] == "postgres"
    assert health["retrieval"] == "hybrid_hash"

    # Persistence: wipe memory via reload from Postgres; answer still cites policy
    reload = httpx.post(
        f"{BASE}/v1/admin/reload",
        headers={"Authorization": "Bearer admin-key"},
        timeout=5.0,
    )
    assert reload.status_code == 200
    assert reload.json()["chunks"] >= 1
    ask2 = httpx.post(
        f"{BASE}/v1/ask",
        headers=AUTH,
        json={"question": "refund policy days"},
        timeout=10.0,
    ).json()
    assert ask2["status"] == "ok"
    assert ask2["citations"][0]["source"] == "policy.md"
