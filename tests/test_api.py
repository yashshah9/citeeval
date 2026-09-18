"""citeeval functional tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from citeeval import api as api_module
from citeeval.evals import run_golden_suite
from citeeval.platform import build_kit
from citeeval.rag import Corpus


@pytest.fixture
def client() -> Iterator[TestClient]:
    api_module.settings.api_keys = "dev-key:demo-tenant"
    api_module.settings.admin_keys = "admin-key"
    api_module.settings.auth_driver = "api_key"
    api_module.settings.audit_driver = "memory"
    api_module.settings.queue_driver = "memory"
    api_module.kit = build_kit(api_module.settings)
    api_module.corpus = Corpus(
        cost_per_ask_usd=api_module.settings.cost_per_ask_usd,
        dense_weight=api_module.settings.dense_weight,
    )
    with TestClient(api_module.app) as test_client:
        yield test_client


AUTH = {"Authorization": "Bearer dev-key"}
ADMIN = {"Authorization": "Bearer admin-key"}


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["retrieval"] == "hybrid_hash"


def test_ask_requires_auth(client: TestClient) -> None:
    assert client.post("/v1/ask", json={"question": "hi"}).status_code == 401


def test_ask_empty_corpus(client: TestClient) -> None:
    resp = client.post("/v1/ask", headers=AUTH, json={"question": "refund policy"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_evidence"
    assert body["citations"] == []
    assert "trace" in body
    assert body["trace"]["hit_count"] == 0


def test_ingest_ask_citations_and_trace(client: TestClient) -> None:
    ing = client.post(
        "/v1/ingest",
        headers=AUTH,
        json={
            "source": "policy.md",
            "text": (
                "Refund policy: customers may request a full refund within 30 days "
                "of purchase if the product is unused."
            ),
        },
    )
    assert ing.status_code == 200
    assert ing.json()["chunks"] >= 1

    ask = client.post(
        "/v1/ask",
        headers=AUTH,
        json={"question": "What is the refund policy window?"},
    )
    body = ask.json()
    assert body["status"] == "ok"
    assert body["citations"]
    assert body["citations"][0]["source"] == "policy.md"
    assert "lexical_score" in body["citations"][0]
    assert "dense_score" in body["citations"][0]
    assert "30 days" in body["answer"]
    assert body["trace"]["retrieval_mode"] == "hybrid_hash"
    assert body["trace"]["estimated_cost_usd"] > 0

    traces = client.get("/v1/traces", headers=AUTH).json()
    assert traces["count"] >= 1
    assert traces["total_estimated_cost_usd"] > 0


def test_eval_pass_and_fail(client: TestClient) -> None:
    client.post(
        "/v1/ingest",
        headers=AUTH,
        json={
            "source": "shipping.md",
            "text": "Standard shipping takes 5 business days to the continental US.",
        },
    )
    resp = client.post(
        "/v1/eval",
        headers=AUTH,
        json={
            "cases": [
                {
                    "question": "How long is standard shipping?",
                    "must_cite_source": "shipping.md",
                    "must_include": "5 business days",
                },
                {
                    "question": "What is the warranty for drones?",
                    "must_cite_source": "warranty.md",
                },
            ]
        },
    )
    body = resp.json()
    assert body["total"] == 2
    assert body["passed"] == 1
    assert body["failed"] == 1


def test_eval_no_evidence_case(client: TestClient) -> None:
    resp = client.post(
        "/v1/eval",
        headers=AUTH,
        json={
            "cases": [
                {
                    "question": "lifetime warranty for drones?",
                    "expect_no_evidence": True,
                }
            ]
        },
    )
    assert resp.json()["passed"] == 1


def test_reset_requires_admin(client: TestClient) -> None:
    assert client.post("/v1/admin/reset", headers=AUTH).status_code == 403
    assert client.post("/v1/admin/reset", headers=ADMIN).status_code == 200


def test_golden_suite_gate() -> None:
    results, rate = run_golden_suite()
    failed = [r for r in results if not r.passed]
    assert failed == [], [(r.case.id, r.reason, r.answer[:80]) for r in failed]
    assert rate == 1.0
