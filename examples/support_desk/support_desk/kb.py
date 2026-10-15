"""A tiny vector search over the help-centre articles, with OpenAI embeddings."""
from __future__ import annotations

import math

from .data import KB_ARTICLES
from .llm import client, tag

EMBED_MODEL = "text-embedding-3-small"


def _embed(texts: list[str]) -> list[list[float]]:
    with tag(feature="kb-search", team="platform"):
        r = client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class KnowledgeBase:
    def __init__(self):
        self.articles = KB_ARTICLES
        self.vectors = _embed([f"{t}\n{body}" for t, body in KB_ARTICLES])     # one batched call

    def search(self, query: str, k: int = 2) -> list[tuple[str, str]]:
        q = _embed([query])[0]
        ranked = sorted(zip(self.vectors, self.articles), key=lambda va: -_cos(q, va[0]))
        return [a for _, a in ranked[:k]]
