from collections import Counter

from conftest import DATA

from fieldassist import corpus
from fieldassist.ingest import parse


def test_seed_is_deterministic_and_matches_committed_data():
    files, golden = corpus.generate()
    for rel, text in files.items():
        assert (DATA / rel).read_text(encoding="utf-8") == text, rel
    assert corpus.load_golden_set(DATA / "golden_set.jsonl") == golden
    assert len(list((DATA / "corpus").rglob("*.md"))) == len(files) == 14


def test_golden_set_has_300_questions_in_seven_slices(golden):
    assert len(golden) == corpus.GOLDEN_SET_SIZE == 300
    assert len({q["id"] for q in golden}) == 300
    assert Counter(q["slice"] for q in golden) == {
        "spec": 45, "torque": 60, "interval": 45, "fault_code": 60, "part_number": 42, "superseded": 24,
        "out_of_scope": 24}
    assert all(not q["answerable"] for q in golden if q["slice"] == "out_of_scope")


def test_every_key_fact_is_stated_by_the_expected_document(golden):
    docs = {parse(p).doc_id: p.read_text(encoding="utf-8") for p in (DATA / "corpus").rglob("*.md")}
    for q in golden:
        if q["answerable"]:
            for fact in q["key_facts"]:
                assert fact.lower() in docs[q["expected_doc"]].lower(), (q["id"], fact)


def test_each_bulletin_changes_exactly_one_paragraph_of_the_section_it_reissues():
    docs = {d.doc_id: d for d in (parse(p) for p in (DATA / "corpus").rglob("*.md"))}
    bulletins = [d for d in docs.values() if d.kind == "bulletin"]
    assert len(bulletins) == 8
    for b in bulletins:
        (target,) = b.supersedes
        old = dict((sid, body) for sid, _, body in docs[target["doc_id"]].sections)[target["section"]]
        new = dict((sid, body) for sid, _, body in b.sections)[target["section"]]
        changed = [(o, n) for o, n in zip(old.split("\n\n"), new.split("\n\n"), strict=True) if o != n]
        assert len(changed) == 1, b.doc_id
