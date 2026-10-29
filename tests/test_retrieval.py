from conftest import cfg

from fieldassist.retrieval import Hit, keyword_search, retrieve, rrf, tsquery_text


def _hit(cid, **kw):
    return Hit(chunk_id=cid, doc_id="D", revision="A", section_id="s", section_title="S", title="T",
               product_line="KR-55", text="", active=True, **kw)


def test_rrf_sums_reciprocal_ranks():
    fused = rrf({"vector": [_hit("a"), _hit("b"), _hit("c")], "keyword": [_hit("c"), _hit("d")]}, k=60, top_k=4)
    scores = {h.chunk_id: round(h.score, 6) for h in fused}
    assert scores["c"] == round(1 / 63 + 1 / 61, 6)
    assert scores["a"] == round(1 / 61, 6)
    assert [h.chunk_id for h in fused][0] == "c"
    assert fused[0].ranks == {"vector": 3, "keyword": 1}


def test_exact_identifier_match_is_placed_first_despite_lower_fused_score():
    lists = {"vector": [_hit("x"), _hit("y")], "keyword": [_hit("e", exact=True), _hit("x"), _hit("y")]}
    fused = rrf(lists, k=60, top_k=3)
    assert fused[0].chunk_id == "e"
    assert fused[1].score > fused[0].score


def test_tsquery_quotes_each_term_and_drops_stopwords():
    assert tsquery_text("What is part 4434-3059?") == "'part' | '4434-3059'"
    assert tsquery_text("What does fault F6580 mean?", only_identifiers=True) == "'f6580'"
    assert tsquery_text("what is it") is None


def test_keyword_search_finds_part_number_as_exact_phrase(pool, indexed):
    hits = keyword_search(pool, "What is part 4434-3059?", cfg("production"))
    assert hits[0].exact and "4434-3059" in hits[0].text
    assert hits[0].section_id == "parts"


def test_revision_filter_hides_superseded_text(pool, embedder, indexed):
    q = "What torque do the airend coupling bolts on the KR-160 take?"
    emb = embedder.embed([q])[0]
    with_filter = retrieve(pool, q, emb, cfg("production"))
    without = retrieve(pool, q, emb, cfg("no-revision-filter"))
    assert all(h.active for h in with_filter)
    assert any(h.doc_id == "SB-2026-004" for h in with_filter)
    assert any(not h.active and h.doc_id == "MAN-KR160" for h in without)


def test_vector_only_loses_fault_code_chunks_hybrid_keeps_them(pool, embedder, indexed):
    q = "The controller is showing F7324. What does that mean?"
    emb = embedder.embed([q])[0]
    assert "F7324" in retrieve(pool, q, emb, cfg("production"))[0].text
    assert not any("F7324" in h.text for h in retrieve(pool, q, emb, cfg("vector-only")))
