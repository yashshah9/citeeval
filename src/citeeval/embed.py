"""Hashing-trick dense vectors — deterministic, no model download."""

from __future__ import annotations

import hashlib
import math
import re
import struct

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "what",
        "how",
        "when",
        "where",
        "who",
        "which",
        "do",
        "does",
        "did",
        "we",
        "our",
        "us",
        "you",
        "your",
        "should",
        "be",
        "with",
        "from",
        "by",
        "at",
        "as",
        "it",
        "this",
        "that",
    }
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def hash_embed(text: str, *, dims: int = 256) -> list[float]:
    """Feature-hash bag-of-tokens into a fixed vector, L2-normalized."""
    vec = [0.0] * dims
    tokens = tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = struct.unpack("<Q", digest)[0] % dims
        sign = 1.0 if digest[0] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b, strict=True)))
