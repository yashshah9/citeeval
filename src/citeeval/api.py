"""HTTP API for citeeval."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from platformkit.protocols import Principal
from pydantic import BaseModel, Field

from citeeval.__version__ import __version__
from citeeval.config import Settings
from citeeval.evals import EvalCase, run_eval
from citeeval.platform import build_kit
from citeeval.rag import Corpus, answer_from_hits

settings = Settings()
kit = build_kit(settings)
corpus = Corpus()

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
        "chunks": len(corpus.chunks),
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
    # optional queue signal for async pipelines later
    kit.queue.enqueue("citeeval.ingest", {"source": body.source, "chunks": len(chunks)})
    return {"source": body.source, "chunks": len(chunks), "chunk_ids": [c.id for c in chunks]}


@app.post("/v1/ask")
def ask(
    body: AskRequest,
    principal: Annotated[Principal, Depends(require_principal)],
) -> dict[str, Any]:
    tenant_id = principal.tenant_id or principal.id
    hits = corpus.search(body.question, top_k=body.top_k)
    answer, citations = answer_from_hits(body.question, hits)
    kit.audit.emit(
        actor=principal.id,
        action="ask.query",
        payload={"question": body.question[:200], "hits": len(hits)},
        tenant_id=tenant_id,
    )
    return {
        "answer": answer,
        "citations": citations,
        "status": "ok" if citations else "no_evidence",
    }


@app.post("/v1/eval")
def eval_endpoint(
    body: EvalRequest,
    principal: Annotated[Principal, Depends(require_principal)],
) -> dict[str, Any]:
    tenant_id = principal.tenant_id or principal.id
    cases = [
        EvalCase(
            question=c.question,
            must_cite_source=c.must_cite_source,
            must_include=c.must_include,
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
        "results": [
            {
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
