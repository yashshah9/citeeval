"""Offline eval runner and golden cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from citeeval.rag import Corpus, answer_from_hits

CASES_PATH = Path(__file__).resolve().parent / "data" / "cases.json"


def resolve_cases_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    bundled = Path(__file__).resolve().parent / "data" / "cases.json"
    if bundled.is_file():
        return bundled
    repo = Path(__file__).resolve().parents[2] / "evals" / "cases.json"
    if repo.is_file():
        return repo
    docker = Path("/app/evals/cases.json")
    if docker.is_file():
        return docker
    raise FileNotFoundError("evals/cases.json not found")


@dataclass
class EvalCase:
    id: str | None = None
    question: str = ""
    must_cite_source: str | None = None
    must_include: str | None = None
    expect_no_evidence: bool = False


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

        if case.expect_no_evidence:
            if citations:
                ok = False
                reasons.append("unexpected_hits")
        else:
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


def load_golden_cases(path: Path | None = None) -> list[EvalCase]:
    raw = json.loads(resolve_cases_path(path).read_text(encoding="utf-8"))
    out: list[EvalCase] = []
    for row in raw["cases"]:
        expect_no = row.get("expect_no_evidence")
        if expect_no is None:
            expect_no = row.get("must_cite_source") is None and row.get("must_include") is None
        out.append(
            EvalCase(
                id=row.get("id"),
                question=row["question"],
                must_cite_source=row.get("must_cite_source"),
                must_include=row.get("must_include"),
                expect_no_evidence=bool(expect_no),
            )
        )
    return out


def seed_corpus_from_golden(path: Path | None = None) -> Corpus:
    raw = json.loads(resolve_cases_path(path).read_text(encoding="utf-8"))
    corpus = Corpus()
    for doc in raw["documents"]:
        corpus.ingest(source=doc["source"], text=doc["text"])
    return corpus


def run_golden_suite(path: Path | None = None) -> tuple[list[EvalResult], float]:
    corpus = seed_corpus_from_golden(path)
    cases = load_golden_cases(path)
    results = run_eval(corpus, cases)
    passed = sum(1 for r in results if r.passed)
    rate = passed / len(results) if results else 0.0
    return results, rate
