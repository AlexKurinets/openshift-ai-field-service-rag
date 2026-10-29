"""Question to cited answer: guard, retrieve, prompt, generate, check citations.

Three refusal paths, all deterministic and in code rather than left to the model: the question
names a model line we hold no documentation for; retrieval is too weak to ground an answer; or the
model's answer cites no passage it was actually given.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from psycopg_pool import ConnectionPool

from .model_client import ModelClient
from .retrieval import Hit, identifiers, retrieve
from .settings import RetrievalConfig

REFUSAL = "I can't find that in the service documentation."
MODEL_NAME = re.compile(r"\b([A-Z]{2,3}-\d{2,4})\b(?!-)")  # KR-110, not SB-2026-004
CITATION = re.compile(r"\[(P\d+)\]")

PROMPTS = {
    "v3": (
        "You answer questions from field-service engineers using only the numbered passages from service "
        "manuals and service bulletins. Quote values exactly as written, with their units. Put the passage "
        "label, such as [P2], after each sentence it supports. A service bulletin replaces the manual section "
        "it reissues. If the passages do not contain the answer, reply exactly: " + REFUSAL
    ),
}


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    reason: str | None
    citations: list[dict] = field(default_factory=list)
    passages: list[Hit] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


class KnownLines:
    """Product lines with documentation in the index, refreshed at most once a minute."""

    def __init__(self, pool: ConnectionPool, ttl: float = 60.0):
        # _at starts as None, not 0.0: time.monotonic() counts from boot, so on a host up for less
        # than ttl seconds a 0.0 start would never look stale and the cache would stay empty.
        self.pool, self.ttl, self._at, self._lines = pool, ttl, None, frozenset()

    def get(self) -> frozenset[str]:
        if self._at is None or time.monotonic() - self._at > self.ttl:
            with self.pool.connection() as conn:
                rows = conn.execute("SELECT DISTINCT product_line FROM documents").fetchall()
            self._lines, self._at = frozenset(r["product_line"] for r in rows), time.monotonic()
        return self._lines


def build_messages(question: str, hits: list[Hit], prompt_version: str) -> list[dict]:
    blocks = [f"[P{i}] {h.title}, revision {h.revision} · {h.section_title}\n{h.text}\n"
              for i, h in enumerate(hits, 1)]
    user = "\n".join(blocks) + f"\nQuestion: {question}"
    return [{"role": "system", "content": PROMPTS[prompt_version]}, {"role": "user", "content": user}]


def answer_question(pool: ConnectionPool, chat: ModelClient, embedder: ModelClient, cfg: RetrievalConfig,
                    question: str, known: KnownLines, product_line: str | None = None) -> Answer:
    t0 = time.monotonic()
    ans = _answer(pool, chat, embedder, cfg, question, known, product_line)
    ans.timings["total"] = time.monotonic() - t0
    return ans


def _answer(pool, chat, embedder, cfg, question, known, product_line) -> Answer:
    t: dict[str, float] = {}
    t0 = time.monotonic()
    unknown = sorted({m for m in MODEL_NAME.findall(question.upper())} - known.get())
    if unknown:
        return Answer(question, REFUSAL, True, f"no documentation for {', '.join(unknown)}", timings=t)

    embedding = embedder.embed([question])[0] if cfg.mode in {"hybrid", "vector"} else None
    t["embed"] = time.monotonic() - t0
    hits = retrieve(pool, question, embedding, cfg, product_line)
    t["retrieve"] = time.monotonic() - t0 - t["embed"]
    if not hits:
        return Answer(question, REFUSAL, True, "no passages retrieved", timings=t)

    best_sim = max((h.similarity for h in hits if h.similarity is not None), default=None)
    ids = set(identifiers(question))
    exact = any(i in h.text.upper() for h in hits for i in ids)
    if best_sim is not None and best_sim < cfg.min_similarity and not exact:
        return Answer(question, REFUSAL, True, f"retrieval too weak ({best_sim:.2f})", passages=hits, timings=t)

    result = chat.chat(build_messages(question, hits, cfg.prompt_version), max_tokens=cfg.max_tokens)
    t["generate"] = result.seconds
    text = result.text.strip()
    if text.startswith(REFUSAL):
        return Answer(question, REFUSAL, True, "model found no answer in the passages", passages=hits, timings=t)

    labels = CITATION.findall(text)
    valid = [lb for lb in labels if 1 <= int(lb[1:]) <= len(hits)]
    if not valid or len(valid) != len(labels):
        return Answer(question, REFUSAL, True, "answer cited no passage it was given", passages=hits, timings=t)
    citations = []
    for lb in dict.fromkeys(valid):
        h = hits[int(lb[1:]) - 1]
        citations.append({"label": lb, "chunk_id": h.chunk_id, "doc_id": h.doc_id, "revision": h.revision,
                          "title": h.title, "section": h.section_title, "active": h.active})
    return Answer(question, text, False, None, citations, hits, t)
