"""In-memory document store with lexical retrieval (no GPU/embeddings required)."""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from threading import Lock

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Chunk:
    id: str
    document_id: str
    source: str
    text: str
    ordinal: int


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Corpus:
    chunks: list[Chunk] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def clear(self) -> None:
        with self._lock:
            self.chunks.clear()

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
                )
                self.chunks.append(chunk)
                created.append(chunk)
        return created

    def search(self, query: str, *, top_k: int = 3) -> list[Hit]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        with self._lock:
            scored: list[Hit] = []
            for chunk in self.chunks:
                c_tokens = tokenize(chunk.text)
                if not c_tokens:
                    continue
                c_set = set(c_tokens)
                overlap = len(q_set & c_set)
                if overlap == 0:
                    continue
                # TF-ish score: overlap / sqrt(len)
                score = overlap / math.sqrt(len(c_set))
                # boost exact phrase
                if query.lower() in chunk.text.lower():
                    score += 1.0
                scored.append(Hit(chunk=chunk, score=score))
            scored.sort(key=lambda h: h.score, reverse=True)
            return scored[: max(1, top_k)]


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
            "excerpt": h.chunk.text[:240],
        }
        for h in hits
    ]
    # Extractive answer: top chunk + question echo (no LLM for MVP determinism)
    top = hits[0].chunk.text
    answer = (
        f"Based on {hits[0].chunk.source}: {top[:500]}"
        if top
        else f"No content for question: {question}"
    )
    return answer, citations
