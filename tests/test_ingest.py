from pathlib import Path

from conftest import DATA

from fieldassist.ingest import MAX_CHUNK_WORDS, chunk, ingest_dir, ingest_documents, parse


def test_chunks_respect_sections_and_size():
    doc = parse(DATA / "corpus" / "manuals" / "MAN-KR110.md")
    chunks = chunk(doc)
    assert {c.section_id for c in chunks} == {s[0] for s in doc.sections}
    for c in chunks:
        body = dict((sid, b) for sid, _, b in doc.sections)[c.section_id]
        assert c.text in body
        assert len(c.text.split()) <= MAX_CHUNK_WORDS or "\n\n" not in c.text
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_superseded_sections_are_inactive_after_full_ingest(pool, indexed):
    assert indexed["documents_indexed"] == 14
    with pool.connection() as conn:
        inactive = {(r["doc_id"], r["section_id"], r["superseded_by"]) for r in conn.execute(
            "SELECT DISTINCT doc_id, section_id, superseded_by FROM chunks WHERE NOT active")}
        expected = {(r["doc_id"], r["section_id"], r["by_doc_id"]) for r in conn.execute(
            "SELECT doc_id, section_id, by_doc_id FROM supersessions")}
    assert inactive == expected
    assert len(expected) == 8


def test_reingest_is_idempotent(pool, embedder, indexed):
    stats = ingest_dir(pool, embedder, DATA / "corpus")
    assert stats["unchanged"] == 14 and stats["documents_indexed"] == 0
    assert stats["active_chunks"] == indexed["active_chunks"]


MANUAL = """---
doc_id: MAN-ZZ9
kind: manual
title: ZZ-9 Service Manual
revision: {rev}
product_line: ZZ-9
published: '{published}'
supersedes: []
---
## Torque values {{#torque-values}}

Pump flange bolts on the ZZ-9: tighten to {nm} Nm.
"""

BULLETIN = """---
doc_id: SB-ZZ9-001
kind: bulletin
title: 'Service Bulletin SB-ZZ9-001: ZZ-9 pump flange torque'
revision: A
product_line: ZZ-9
published: '2026-05-01'
supersedes:
- doc_id: MAN-ZZ9
  section: torque-values
---
## Torque values (reissued) {#torque-values}

Pump flange bolts on the ZZ-9: tighten to 44 Nm.
"""


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


def _active(pool, doc_id):
    with pool.connection() as conn:
        return {r["revision"]: r["active"] for r in conn.execute(
            "SELECT revision, bool_and(active) AS active FROM chunks WHERE doc_id = %s GROUP BY revision", (doc_id,))}


def _cleanup(pool):
    with pool.connection() as conn:
        conn.execute("DELETE FROM documents WHERE product_line = 'ZZ-9'")
        conn.execute("DELETE FROM supersessions WHERE by_doc_id = 'SB-ZZ9-001'")


def test_bulletin_ingested_before_its_manual_still_supersedes_it(pool, embedder, indexed, tmp_path):
    try:
        ingest_documents(pool, embedder, [parse(_write(tmp_path, "b.md", BULLETIN))])
        manual = MANUAL.format(rev="C", published="2025-03-14", nm=40)
        ingest_documents(pool, embedder, [parse(_write(tmp_path, "m.md", manual))])
        assert _active(pool, "MAN-ZZ9") == {"C": False}
        assert _active(pool, "SB-ZZ9-001") == {"A": True}
    finally:
        _cleanup(pool)


def test_newer_manual_revision_replaces_the_older_one(pool, embedder, indexed, tmp_path):
    try:
        old = parse(_write(tmp_path, "c.md", MANUAL.format(rev="C", published="2025-03-14", nm=40)))
        new = parse(_write(tmp_path, "d.md", MANUAL.format(rev="D", published="2026-02-01", nm=42)))
        ingest_documents(pool, embedder, [new, old])
        assert _active(pool, "MAN-ZZ9") == {"C": False, "D": True}
    finally:
        _cleanup(pool)
