"""Build corpus with optional durable backend."""

from __future__ import annotations

from citeeval.config import Settings
from citeeval.rag import Corpus
from citeeval.store import build_backend


def build_corpus(settings: Settings) -> Corpus:
    backend = build_backend(settings.corpus_driver, postgres_dsn=settings.postgres_dsn)
    corpus = Corpus(
        cost_per_ask_usd=settings.cost_per_ask_usd,
        dense_weight=settings.dense_weight,
        backend=backend,
    )
    corpus.load()
    return corpus
