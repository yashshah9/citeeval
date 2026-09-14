"""Offline eval runner for citeeval."""

from __future__ import annotations

from dataclasses import dataclass

from citeeval.rag import Corpus, answer_from_hits


@dataclass
class EvalCase:
    question: str
    must_cite_source: str | None = None
    must_include: str | None = None


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    answer: str
    citations: list[dict[str, object]]
    reason: str


def run_eval(corpus: Corpus, cases: list[EvalCase], *, top_k: int = 3) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        hits = corpus.search(case.question, top_k=top_k)
        answer, citations = answer_from_hits(case.question, hits)
        reasons: list[str] = []
        ok = True
        if not citations:
            ok = False
            reasons.append("no_citations")
        if case.must_cite_source:
            sources = {str(c.get("source")) for c in citations}
            if case.must_cite_source not in sources:
                ok = False
                reasons.append("missing_source")
        if case.must_include and case.must_include.lower() not in answer.lower():
            ok = False
            reasons.append("missing_answer_text")
        results.append(
            EvalResult(
                case=case,
                passed=ok,
                answer=answer,
                citations=citations,
                reason=",".join(reasons) if reasons else "ok",
            )
        )
    return results
