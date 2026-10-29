"""Nightly ingestion: parse, chunk by section, embed, index, then recompute which text is current.

In the cluster this is the ingest pipeline (pipelines/ingest_pipeline.py) on a nightly recurring
run. Documents arrive in no guaranteed order, so whether a chunk is active is never decided while
inserting it; after every run one UPDATE recomputes it from the latest revision of each document
and from the supersessions table. A bulletin ingested before the manual it amends still wins.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from psycopg_pool import ConnectionPool

from .db import vector_literal
from .model_client import ModelClient

MAX_CHUNK_WORDS = 120
HEADING = re.compile(r"^## (.+?) \{#([a-z0-9-]+)\}\s*$", re.M)


@dataclass
class Document:
    doc_id: str
    revision: str
    kind: str
    title: str
    product_line: str
    published: str
    supersedes: list[dict]
    sections: list[tuple[str, str, str]]  # (section_id, title, body)
    content_hash: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    revision: str
    section_id: str
    section_title: str
    ordinal: int
    product_line: str
    text: str
    embed_text: str = field(repr=False, default="")


def parse(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path}: no front matter")
    _, fm, body = raw.split("---\n", 2)
    meta = yaml.safe_load(fm)
    heads = list(HEADING.finditer(body))
    sections = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        sections.append((m.group(2), m.group(1), body[m.end():end].strip()))
    return Document(
        doc_id=meta["doc_id"], revision=str(meta["revision"]), kind=meta["kind"], title=meta["title"],
        product_line=meta["product_line"], published=str(meta["published"]),
        supersedes=meta.get("supersedes") or [], sections=sections,
        content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


def chunk(doc: Document, max_words: int = MAX_CHUNK_WORDS) -> list[Chunk]:
    """Pack whole paragraphs into chunks of at most max_words; never split a paragraph or cross a section."""
    out: list[Chunk] = []
    for section_id, title, body in doc.sections:
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        groups: list[list[str]] = []
        for p in paras:
            words = len(p.split())
            if groups and sum(len(x.split()) for x in groups[-1]) + words <= max_words:
                groups[-1].append(p)
            else:
                groups.append([p])
        for ordinal, g in enumerate(groups):
            key = f"{doc.doc_id}:{doc.revision}:{section_id}:{ordinal}"
            text = "\n\n".join(g)
            out.append(Chunk(
                chunk_id="c" + hashlib.sha1(key.encode()).hexdigest()[:10], doc_id=doc.doc_id,
                revision=doc.revision, section_id=section_id, section_title=title, ordinal=ordinal,
                product_line=doc.product_line, text=text, embed_text=f"{doc.title} · {title}\n{text}",
            ))
    return out


RECOMPUTE_ACTIVE = """
WITH latest AS (
    SELECT DISTINCT ON (doc_id) doc_id, revision FROM documents ORDER BY doc_id, published DESC, revision DESC
), state AS (
    SELECT c.chunk_id,
           (c.revision = l.revision AND s.by_doc_id IS NULL) AS active,
           s.by_doc_id AS superseded_by
    FROM chunks c
    JOIN latest l ON l.doc_id = c.doc_id
    LEFT JOIN LATERAL (
        SELECT min(sp.by_doc_id) AS by_doc_id FROM supersessions sp
        JOIN documents d ON d.doc_id = sp.by_doc_id
        WHERE sp.doc_id = c.doc_id AND sp.section_id = c.section_id
    ) s ON true
)
UPDATE chunks c SET active = state.active, superseded_by = state.superseded_by
FROM state
WHERE state.chunk_id = c.chunk_id
  AND (c.active IS DISTINCT FROM state.active OR c.superseded_by IS DISTINCT FROM state.superseded_by)
"""


def ingest_documents(pool: ConnectionPool, embedder: ModelClient, docs: list[Document]) -> dict[str, int]:
    stats = {"documents_seen": len(docs), "documents_indexed": 0, "chunks_indexed": 0, "unchanged": 0}
    with pool.connection() as conn:
        existing = {(r["doc_id"], r["revision"]): r["content_hash"]
                    for r in conn.execute("SELECT doc_id, revision, content_hash FROM documents")}
    for doc in docs:
        if existing.get((doc.doc_id, doc.revision)) == doc.content_hash:
            stats["unchanged"] += 1
            continue
        chunks = chunk(doc)
        vectors = embedder.embed([c.embed_text for c in chunks])
        with pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM documents WHERE doc_id = %s AND revision = %s", (doc.doc_id, doc.revision))
            conn.execute("DELETE FROM supersessions WHERE by_doc_id = %s", (doc.doc_id,))
            conn.execute(
                "INSERT INTO documents (doc_id, revision, kind, title, product_line, published, content_hash) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (doc.doc_id, doc.revision, doc.kind, doc.title, doc.product_line, doc.published, doc.content_hash))
            for s in doc.supersedes:
                conn.execute("INSERT INTO supersessions (by_doc_id, doc_id, section_id) VALUES (%s, %s, %s) "
                             "ON CONFLICT DO NOTHING", (doc.doc_id, s["doc_id"], s["section"]))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, revision, section_id, section_title, ordinal, "
                    "product_line, text, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                    [(c.chunk_id, c.doc_id, c.revision, c.section_id, c.section_title, c.ordinal, c.product_line,
                      c.text, vector_literal(v)) for c, v in zip(chunks, vectors, strict=True)])
        stats["documents_indexed"] += 1
        stats["chunks_indexed"] += len(chunks)
    with pool.connection() as conn:
        conn.execute(RECOMPUTE_ACTIVE)
        counts = conn.execute(
            "SELECT count(*) FILTER (WHERE active) AS active, count(*) AS total FROM chunks").fetchone()
    stats["active_chunks"] = counts["active"]
    stats["total_chunks"] = counts["total"]
    return stats


def ingest_dir(pool: ConnectionPool, embedder: ModelClient, corpus_dir: Path) -> dict[str, int]:
    docs = [parse(p) for p in sorted(corpus_dir.rglob("*.md"))]
    return ingest_documents(pool, embedder, docs)
