"""Hybrid retrieval: pgvector cosine search and Postgres full-text search, fused by reciprocal rank.

Dense embeddings blur identifiers (a part number embeds like every other part number), and
full-text search misses paraphrases, so each retriever returns its top `candidates_per_retriever`
and the lists are merged with reciprocal rank fusion: score = sum of 1 / (rrf_k + rank). RRF uses
ranks only, so the two retrievers' incomparable scores never have to be calibrated against each other.

One exception to pure RRF: a chunk containing a part number or fault code from the question is always
placed first. Without it, a chunk ranked first by keyword alone (1/61) loses to chunks that sit
lower on both lists (1/63 + 1/62), and those are exactly the near-identical fault-code chunks that
dense retrieval can't separate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from psycopg_pool import ConnectionPool

from .db import vector_literal
from .settings import RetrievalConfig

WORD = re.compile(r"[A-Za-z0-9]+(?:[-.,/][A-Za-z0-9]+)*")
QUERY_STOPWORDS = frozenset("""
a an and are as at be by can do does for from how i in is it of on or should the to what when where which
who why with you your give gives much many belong belongs mean means
""".split())


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    revision: str
    section_id: str
    section_title: str
    title: str
    product_line: str
    text: str
    active: bool
    similarity: float | None = None
    keyword_rank: float | None = None
    exact: bool = False
    ranks: dict[str, int] = field(default_factory=dict)
    score: float = 0.0


def is_identifier(word: str) -> bool:
    """Part numbers and fault codes: four or more digits (4471-0213, F2140)."""
    return sum(c.isdigit() for c in word) >= 4


def identifiers(question: str) -> list[str]:
    return [w.strip(".,").upper() for w in WORD.findall(question) if is_identifier(w)]


def tsquery_text(question: str, only_identifiers: bool = False) -> str | None:
    """OR of the question's terms for to_tsquery. Each term is quoted, so an identifier such as
    4471-0213 is parsed by the same text parser that indexed the chunk and must match as a phrase."""
    terms = []
    for w in WORD.findall(question):
        lw = w.lower().strip(".,")
        if only_identifiers and not is_identifier(lw):
            continue
        if lw and lw not in QUERY_STOPWORDS and lw not in terms:
            terms.append(lw)
    if not terms:
        return None
    return " | ".join("'" + t.replace("'", "") + "'" for t in terms)


SELECT_COLS = """c.chunk_id, c.doc_id, c.revision, c.section_id, c.section_title, d.title, c.product_line,
                 c.text, c.active"""


def vector_search(pool: ConnectionPool, embedding: list[float], cfg: RetrievalConfig,
                  product_line: str | None = None) -> list[Hit]:
    where = ["true"]
    params: list = [vector_literal(embedding)]
    if cfg.active_only:
        where.append("c.active")
    if product_line:
        where.append("c.product_line = %s")
        params.append(product_line)
    params += [vector_literal(embedding), cfg.candidates_per_retriever]
    sql = f"""SELECT {SELECT_COLS}, 1 - (c.embedding <=> %s::vector) AS similarity
              FROM chunks c JOIN documents d USING (doc_id, revision)
              WHERE {' AND '.join(where)}
              ORDER BY c.embedding <=> %s::vector LIMIT %s"""
    with pool.connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [Hit(**{k: r[k] for k in r if k != "similarity"}, similarity=float(r["similarity"])) for r in rows]


def keyword_search(pool: ConnectionPool, question: str, cfg: RetrievalConfig,
                   product_line: str | None = None) -> list[Hit]:
    tsq = tsquery_text(question)
    if not tsq:
        return []
    where = ["c.tsv @@ q.query"]
    params: list = [tsq, tsquery_text(question, only_identifiers=True)]
    if cfg.active_only:
        where.append("c.active")
    if product_line:
        where.append("c.product_line = %s")
        params.append(product_line)
    params.append(cfg.candidates_per_retriever)
    # A chunk containing the exact part number or fault code outranks any amount of shared vocabulary.
    sql = f"""SELECT {SELECT_COLS}, ts_rank_cd(c.tsv, q.query) AS keyword_rank,
                     coalesce(c.tsv @@ i.ids, false) AS exact
              FROM chunks c JOIN documents d USING (doc_id, revision),
                   to_tsquery('english', %s) AS q(query)
                   LEFT JOIN LATERAL (SELECT to_tsquery('english', %s::text) AS ids) i ON true
              WHERE {' AND '.join(where)}
              ORDER BY exact DESC, keyword_rank DESC, c.chunk_id LIMIT %s"""
    with pool.connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [Hit(**{k: r[k] for k in r if k != "keyword_rank"}, keyword_rank=float(r["keyword_rank"])) for r in rows]


def rrf(lists: dict[str, list[Hit]], k: int, top_k: int) -> list[Hit]:
    merged: dict[str, Hit] = {}
    for name, hits in lists.items():
        for rank, h in enumerate(hits, 1):
            cur = merged.setdefault(h.chunk_id, h)
            if h.similarity is not None:
                cur.similarity = h.similarity
            if h.keyword_rank is not None:
                cur.keyword_rank = h.keyword_rank
            cur.exact = cur.exact or h.exact
            cur.ranks[name] = rank
            cur.score += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda h: (not h.exact, -h.score, h.chunk_id))[:top_k]


def retrieve(pool: ConnectionPool, question: str, embedding: list[float] | None, cfg: RetrievalConfig,
             product_line: str | None = None) -> list[Hit]:
    lists: dict[str, list[Hit]] = {}
    if cfg.mode in {"hybrid", "vector"}:
        lists["vector"] = vector_search(pool, embedding, cfg, product_line)
    if cfg.mode in {"hybrid", "keyword"}:
        lists["keyword"] = keyword_search(pool, question, cfg, product_line)
    return rrf(lists, cfg.rrf_k, cfg.top_k)
