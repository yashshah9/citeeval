# Demo script (~5 minutes)

## Setup

```bash
uv run citeeval serve
# or: docker compose up --build -d
curl -s localhost:8091/health | jq
```

## 1. Ingest + ask with citations (2 min)

```bash
curl -s -X POST localhost:8091/v1/ingest \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"source":"policy.md","text":"Refund policy: full refund within 30 days if unused."}'

curl -s -X POST localhost:8091/v1/ask \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund window?"}' | jq
```

Show: citations with `lexical_score` / `dense_score`, and `trace.latency_ms` + `estimated_cost_usd`.

## 2. Traces (1 min)

```bash
curl -s localhost:8091/v1/traces -H "Authorization: Bearer dev-key" | jq
```

## 3. No-evidence path (30s)

```bash
curl -s -X POST localhost:8091/v1/ask \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"question":"drone lifetime warranty?"}' | jq .status
```

Expect `no_evidence`.

## 4. Eval gate (1 min)

```bash
citeeval eval --min-pass-rate 0.8 --baseline evals/baseline.json
```

Show pass_rate and **BASELINE OK**. Mention CI fails on both pass-rate drops and case-level regressions (or top-source drift).
