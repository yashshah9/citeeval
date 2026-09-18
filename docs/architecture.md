# Architecture

## Pipeline

```
ingest ──► chunk + hash-embed ──► corpus
ask    ──► hybrid retrieve (lexical + dense) ──► extractive answer + citations
       ──► AskTrace (latency, cost estimate, scores)
eval   ──► golden cases gate (CI)
```

## Retrieval

Hybrid score:

`score = (1 - w) * lexical + w * dense`

- **lexical**: token overlap (+ phrase boost)
- **dense**: hashing-trick embeddings (`citeeval.embed`) — deterministic, no model download

Swap `hash_embed` for real embeddings later without changing the API.

## Observability

Every `/v1/ask` records an `AskTrace`:

- `latency_ms`
- `hit_count` / `top_score`
- `estimated_cost_usd` (flat extractive estimate today)
- `retrieval_mode`

List recent traces via `GET /v1/traces`.

## Corpus durability

| Driver | Behavior |
|--------|----------|
| `memory` (default) | process-local chunks |
| `postgres` | `citeeval_chunks` table; load on boot; `POST /v1/admin/reload` rehydrates |

Compose sets `CITEEVAL_CORPUS_DRIVER=postgres`.

## Eval promotion gate

```bash
citeeval eval --min-pass-rate 0.8 --baseline evals/baseline.json
# refresh after intentional improvements:
citeeval eval --write-baseline evals/baseline.json
```

Golden corpus + cases live in `evals/cases.json` (bundled under `citeeval/data/`).
CI fails when the pass rate drops **or** a previously-passing case regresses / changes top citation source.
