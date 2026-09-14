# citeeval

Production-shaped **RAG** with citations, offline **evals**, and pluggable auth/audit/queue via [platformkit](../platformkit).

MVP retrieval is deterministic lexical search (no paid embeddings) so CI/docker stays reliable.

## Docker

```bash
cd citeeval
docker compose up --build -d redis postgres citeeval
docker compose run --rm integration
```

## Local tests

```bash
uv venv --python 3.12
uv pip install -e ../platformkit -e ".[dev]"
uv run pytest tests/test_api.py -v
```

## API

- `POST /v1/ingest` — add documents
- `POST /v1/ask` — question + citations
- `POST /v1/eval` — golden-case eval gate
- `POST /v1/admin/reset` — admin only
