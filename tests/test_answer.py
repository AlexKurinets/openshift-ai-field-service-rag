import pytest
from conftest import cfg

from fieldassist.answer import REFUSAL, KnownLines, answer_question
from fieldassist.model_client import ChatResult


class FakeChat:
    def __init__(self, text=None):
        self.text, self.calls = text, 0

    def chat(self, messages, max_tokens=400):
        self.calls += 1
        if self.text is None:
            raise AssertionError("the model should not have been called")
        return ChatResult(self.text, 0, 0, 0.0)


@pytest.fixture
def known(pool, indexed):
    return KnownLines(pool)


def ask(pool, chat, embedder, known, q, name="production"):
    return answer_question(pool, chat, embedder, cfg(name), q, known)


def test_unknown_model_is_refused_before_retrieval_or_generation(pool, embedder, known):
    fake = FakeChat()
    ans = ask(pool, fake, embedder, known, "What torque do the motor mount bolts on the KR-300 take?")
    assert ans.refused and "KR-300" in ans.reason and fake.calls == 0 and ans.passages == []


def test_bulletin_number_is_not_mistaken_for_a_model_name(pool, chat, embedder, known):
    ans = ask(pool, chat, embedder, known, "What does SB-2026-004 change about the KR-160 airend coupling bolts?")
    assert not ans.refused
    assert ans.citations[0]["doc_id"] == "SB-2026-004"


def test_off_topic_question_is_refused_on_weak_retrieval(pool, embedder, known):
    ans = ask(pool, FakeChat(), embedder, known, "How do I book a meeting room for Thursday?")
    assert ans.refused and ans.reason.startswith("retrieval too weak")


def test_answer_without_citation_is_not_shown(pool, embedder, known):
    ans = ask(pool, FakeChat("Tighten them to 105 Nm."), embedder, known,
              "What torque do the airend coupling bolts on the KR-160 take?")
    assert ans.refused and ans.text == REFUSAL and "cited no passage" in ans.reason


def test_citation_to_a_passage_that_was_never_given_is_rejected(pool, embedder, known):
    ans = ask(pool, FakeChat("Tighten to 105 Nm [P1] [P9]."), embedder, known,
              "What torque do the airend coupling bolts on the KR-160 take?")
    assert ans.refused


def test_superseded_value_comes_from_the_bulletin(pool, chat, embedder, known):
    ans = ask(pool, chat, embedder, known, "What torque do the airend coupling bolts on the KR-160 take?")
    assert not ans.refused and "105 Nm" in ans.text and "95 Nm in a cross pattern" not in ans.text
    assert {c["doc_id"] for c in ans.citations} == {"SB-2026-004"}
    assert all(c["active"] for c in ans.citations)
