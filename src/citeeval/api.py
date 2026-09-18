"""HTTP API for citeeval."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from platformkit.protocols import Principal
from pydantic import BaseModel, Field

from citeeval.__version__ import __version__
from citeeval.config import Settings
from citeeval.corpus_factory import build_corpus
from citeeval.evals import EvalCase, run_eval
from citeeval.platform import build_kit

settings = Settings()
kit = build_kit(settings)
corpus = build_corpus(settings)

app = FastAPI(title="citeeval", version=__version__)


class IngestRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=500)
    text: str = Field(..., min_length=1, max_length=200_000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)


class EvalCaseModel(BaseModel):
    question: str
    must_cite_source: str | None = None
    must_include: str | None = None
    expect_no_evidence: bool = False
    id: str | None = None


class EvalRequest(BaseModel):
    cases: list[EvalCaseModel] = Field(..., min_length=1)


def _bearer_token(authorization: Annotated[str | None, Header()] = None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    return authorization.split(" ", 1)[1].strip()


def require_principal(token: Annotated[str, Depends(_bearer_token)]) -> Principal:
    principal = kit.auth.authenticate(token)
    if principal is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return principal


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "auth": settings.auth_driver,
        "audit": settings.audit_driver,
        "queue": settings.queue_driver,
        "corpus": settings.corpus_driver,
        "chunks": len(corpus.chunks),
        "traces": len(corpus.traces),
        "retrieval": "hybrid_hash",
    }


@app.post("/v1/ingest")
def ingest(
    body: IngestRequest,
    principal: Annotated[Principal, Depends(require_principal)],
) -> dict[str, Any]:
    tenant_id = principal.tenant_id or principal.id
    chunks = corpus.ingest(source=body.source, text=body.text)
    kit.audit.emit(
        actor=principal.id,
        action="doc.ingest",
        payload={"source": body.source, "chunks": len(chunks)},
        tenant_id=tenant_id,
    )
    kit.queue.enqueue("citeeval.ingest", {"source": body.source, "chunks": len(chunks)})
    return {"source": body.source, "chunks": len(chunks), "chunk_ids": [c.id for c in chunks]}


@app.post("/v1/ask")
def ask(
    body: AskRequest,
    principal: Annotated[Principal, Depends(require_principal)],
) -> dict[str, Any]:
    tenant_id = principal.tenant_id or principal.id
    answer, citations, trace = corpus.ask(body.question, top_k=body.top_k)
    kit.audit.emit(
        actor=principal.id,
        action="ask.query",
        payload={
            "question": body.question[:200],
            "hits": trace.hit_count,
            "latency_ms": trace.latency_ms,
            "cost_usd": trace.estimated_cost_usd,
        },
        tenant_id=tenant_id,
    )
    return {
        "answer": answer,
        "citations": citations,
        "status": "ok" if citations else "no_evidence",
        "trace": {
            "latency_ms": trace.latency_ms,
            "hit_count": trace.hit_count,
            "top_score": trace.top_score,
            "estimated_cost_usd": trace.estimated_cost_usd,
            "retrieval_mode": trace.retrieval_mode,
            "chunk_ids": trace.chunk_ids,
        },
    }


@app.get("/v1/traces")
def list_traces(
    principal: Annotated[Principal, Depends(require_principal)],
    limit: int = 20,
) -> dict[str, Any]:
    del principal
    limit = max(1, min(limit, 100))
    items = list(reversed(corpus.traces[-limit:]))
    total_cost = sum(t.estimated_cost_usd for t in corpus.traces)
    return {
        "count": len(items),
        "total_estimated_cost_usd": round(total_cost, 6),
        "traces": [
            {
                "question": t.question,
                "latency_ms": t.latency_ms,
                "hit_count": t.hit_count,
                "top_score": t.top_score,
                "estimated_cost_usd": t.estimated_cost_usd,
                "retrieval_mode": t.retrieval_mode,
            }
            for t in items
        ],
    }


@app.post("/v1/eval")
def eval_endpoint(
    body: EvalRequest,
    principal: Annotated[Principal, Depends(require_principal)],
) -> dict[str, Any]:
    tenant_id = principal.tenant_id or principal.id
    cases = [
        EvalCase(
            id=c.id,
            question=c.question,
            must_cite_source=c.must_cite_source,
            must_include=c.must_include,
            expect_no_evidence=c.expect_no_evidence,
        )
        for c in body.cases
    ]
    results = run_eval(corpus, cases)
    passed = sum(1 for r in results if r.passed)
    kit.audit.emit(
        actor=principal.id,
        action="eval.run",
        payload={"total": len(results), "passed": passed},
        tenant_id=tenant_id,
    )
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": [
            {
                "id": r.case.id,
                "question": r.case.question,
                "passed": r.passed,
                "reason": r.reason,
                "answer": r.answer,
                "citations": r.citations,
            }
            for r in results
        ],
    }


@app.post("/v1/admin/reset")
def reset_corpus(
    principal: Annotated[Principal, Depends(require_principal)],
) -> dict[str, str]:
    if "admin" not in principal.roles:
        raise HTTPException(status_code=403, detail="admin required")
    corpus.clear()
    return {"status": "cleared"}


@app.post("/v1/admin/reload")
def reload_corpus(
    principal: Annotated[Principal, Depends(require_principal)],
) -> dict[str, Any]:
    """Drop in-memory chunks and rehydrate from durable backend (if configured)."""
    if "admin" not in principal.roles:
        raise HTTPException(status_code=403, detail="admin required")
    if corpus.backend is None:
        return {
            "status": "noop",
            "corpus": settings.corpus_driver,
            "chunks": len(corpus.chunks),
            "detail": "no durable backend; memory corpus left intact",
        }
    with corpus._lock:
        corpus.chunks.clear()
        corpus.traces.clear()
    loaded = corpus.load()
    return {
        "status": "reloaded",
        "corpus": settings.corpus_driver,
        "chunks": loaded,
    }
