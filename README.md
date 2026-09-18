# citeeval

Production-shaped **RAG** with citations, hybrid retrieval, offline **evals**, and cost/latency traces.

Uses [platformkit](https://github.com/yashshah9/platformkit) for pluggable auth/audit/queues.

## Quick start

```bash
uv venv --python 3.12
uv pip install -e ../platformkit -e ".[dev]"
uv run pytest tests/test_api.py -q
uv run citeeval eval --min-pass-rate 0.8 --baseline evals/baseline.json
uv run citeeval serve   # :8091
```

## Docker

```bash
docker compose up --build -d redis postgres citeeval
docker compose run --rm integration
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/ingest` | add documents |
| POST | `/v1/ask` | question + citations + trace |
| GET | `/v1/traces` | recent ask traces + cost rollup |
| POST | `/v1/eval` | run eval cases against live corpus |
| POST | `/v1/admin/reset` | clear corpus (admin) |

## Docs

- [Architecture](docs/architecture.md)
- [Demo script](docs/demo.md)
