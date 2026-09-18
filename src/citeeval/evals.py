"""Offline eval runner, golden cases, and regression baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def resolve_baseline_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    bundled = Path(__file__).resolve().parent / "data" / "baseline.json"
    if bundled.is_file():
        return bundled
    repo = Path(__file__).resolve().parents[2] / "evals" / "baseline.json"
    if repo.is_file():
        return repo
    docker = Path("/app/evals/baseline.json")
    if docker.is_file():
        return docker
    raise FileNotFoundError("evals/baseline.json not found")


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

    @property
    def case_id(self) -> str:
        return self.case.id or self.case.question[:64]

    @property
    def top_source(self) -> str | None:
        if not self.citations:
            return None
        return str(self.citations[0].get("source"))

    @property
    def top_score(self) -> float | None:
        if not self.citations:
            return None
        score = self.citations[0].get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return None


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
        out.append(
            EvalCase(
                id=row.get("id"),
                question=row["question"],
                must_cite_source=row.get("must_cite_source"),
                must_include=row.get("must_include"),
                expect_no_evidence=bool(row.get("expect_no_evidence", False)),
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


def snapshot_from_results(results: list[EvalResult], *, pass_rate: float) -> dict[str, Any]:
    return {
        "pass_rate": round(pass_rate, 4),
        "cases": {
            r.case_id: {
                "passed": r.passed,
                "reason": r.reason,
                "top_source": r.top_source,
                "top_score": round(r.top_score, 4) if r.top_score is not None else None,
            }
            for r in results
            if r.case_id
        },
    }


def write_baseline(path: Path, results: list[EvalResult], *, pass_rate: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot_from_results(results, pass_rate=pass_rate), indent=2) + "\n",
        encoding="utf-8",
    )


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    raw: object = json.loads(resolve_baseline_path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("baseline root must be an object")
    return raw


def compare_to_baseline(
    results: list[EvalResult],
    baseline: dict[str, Any],
) -> list[str]:
    """Return human-readable regression messages (empty = ok)."""
    regressions: list[str] = []
    prior_cases = baseline.get("cases", {})
    if not isinstance(prior_cases, dict):
        return ["baseline.cases missing or invalid"]

    current = {r.case_id: r for r in results if r.case_id}
    for case_id, prior in prior_cases.items():
        if not isinstance(prior, dict):
            continue
        now = current.get(str(case_id))
        if now is None:
            regressions.append(f"{case_id}: missing from current suite")
            continue
        if prior.get("passed") is True and not now.passed:
            regressions.append(f"{case_id}: was pass, now fail ({now.reason})")
            continue
        prior_source = prior.get("top_source")
        if (
            prior.get("passed") is True
            and prior_source
            and now.top_source
            and prior_source != now.top_source
        ):
            regressions.append(
                f"{case_id}: top_source changed {prior_source!r} → {now.top_source!r}"
            )
    return regressions
