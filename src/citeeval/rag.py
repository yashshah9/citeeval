"""In-memory corpus with hybrid lexical + dense retrieval."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from citeeval.embed import cosine, hash_embed, tokenize


@dataclass
class Chunk:
    id: str
    document_id: str
    source: str
    text: str
    ordinal: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    lexical_score: float = 0.0
    dense_score: float = 0.0


@dataclass
class AskTrace:
    question: str
    latency_ms: int
    hit_count: int
    top_score: float | None
    estimated_cost_usd: float
    retrieval_mode: str
    chunk_ids: list[str]


@dataclass
class Corpus:
    chunks: list[Chunk] = field(default_factory=list)
    traces: list[AskTrace] = field(default_factory=list)
    # ponytail: flat USD estimate for extractive answers; replace with real LLM metering
    cost_per_ask_usd: float = 0.0001
    dense_weight: float = 0.45
    backend: Any | None = None
    _lock: Lock = field(default_factory=Lock)

    def load(self) -> int:
        """Hydrate in-memory chunks from durable backend. Returns loaded count."""
        if self.backend is None:
            return 0
        loaded = self.backend.load_all()
        with self._lock:
            self.chunks = loaded
        return len(loaded)

    def clear(self) -> None:
        with self._lock:
            self.chunks.clear()
            self.traces.clear()
        if self.backend is not None:
            self.backend.clear()

    def ingest(self, *, source: str, text: str, chunk_size: int = 400) -> list[Chunk]:
        doc_id = str(uuid.uuid4())
        parts = _chunk_text(text, chunk_size=chunk_size)
        created: list[Chunk] = []
        with self._lock:
            for i, part in enumerate(parts):
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    source=source,
                    text=part,
                    ordinal=i,
                    embedding=hash_embed(part),
                )
                self.chunks.append(chunk)
                created.append(chunk)
        if self.backend is not None and created:
            self.backend.save_chunks(created)
        return created

    def search(self, query: str, *, top_k: int = 3) -> list[Hit]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        q_emb = hash_embed(query)
        with self._lock:
            scored: list[Hit] = []
            for chunk in self.chunks:
                c_tokens = tokenize(chunk.text)
                if not c_tokens:
                    continue
                c_set = set(c_tokens)
                overlap = len(q_set & c_set)
                lexical = 0.0
                if overlap:
                    lexical = overlap / math.sqrt(len(c_set))
                    if query.lower() in chunk.text.lower():
                        lexical += 1.0
                dense = cosine(q_emb, chunk.embedding)
                # hybrid: require meaningful lexical overlap (ignore weak dense-only noise)
                if lexical < 0.35:
                    continue
                score = (1.0 - self.dense_weight) * lexical + self.dense_weight * dense
                scored.append(
                    Hit(
                        chunk=chunk,
                        score=score,
                        lexical_score=lexical,
                        dense_score=dense,
                    )
                )
            scored.sort(key=lambda h: h.score, reverse=True)
            return scored[: max(1, top_k)]

    def record_trace(self, trace: AskTrace) -> None:
        with self._lock:
            self.traces.append(trace)
            # keep last 500
            if len(self.traces) > 500:
                self.traces = self.traces[-500:]

    def ask(
        self, question: str, *, top_k: int = 3
    ) -> tuple[str, list[dict[str, object]], AskTrace]:
        started = time.perf_counter()
        hits = self.search(question, top_k=top_k)
        answer, citations = answer_from_hits(question, hits)
        latency_ms = int((time.perf_counter() - started) * 1000)
        trace = AskTrace(
            question=question[:200],
            latency_ms=latency_ms,
            hit_count=len(hits),
            top_score=round(hits[0].score, 4) if hits else None,
            estimated_cost_usd=self.cost_per_ask_usd,
            retrieval_mode="hybrid_hash",
            chunk_ids=[h.chunk.id for h in hits],
        )
        self.record_trace(trace)
        return answer, citations, trace


def _chunk_text(text: str, *, chunk_size: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end
    return chunks


def answer_from_hits(question: str, hits: list[Hit]) -> tuple[str, list[dict[str, object]]]:
    if not hits:
        return (
            "I could not find supporting evidence in the corpus.",
            [],
        )
    citations = [
        {
            "source": h.chunk.source,
            "chunk_id": h.chunk.id,
            "score": round(h.score, 4),
            "lexical_score": round(h.lexical_score, 4),
            "dense_score": round(h.dense_score, 4),
            "excerpt": h.chunk.text[:240],
        }
        for h in hits
    ]
    top = hits[0].chunk.text
    answer = (
        f"Based on {hits[0].chunk.source}: {top[:500]}"
        if top
        else f"No content for question: {question}"
    )
    return answer, citations
