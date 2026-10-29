"""Stand-in for Granite 3.3 8B Instruct: an extractive reader that follows the gateway's prompt.

It reads the numbered passages and the question out of the user message, picks the sentence that
shares the most content words with the question, and answers with that sentence and its passage
label. Model numbers count double and part numbers or fault codes triple, a sentence carrying a value
the question doesn't already contain gets a small bonus, and the shorter sentence wins a tie. When
nothing overlaps enough it gives the refusal the system prompt asks for. Deterministic, so golden-set
runs reproduce exactly; the real model paraphrases, which is why the gateway checks citations rather
than trusting them.
"""

from __future__ import annotations

import re

from .text import is_identifier, tokens

PASSAGE = re.compile(r"^\[(P\d+)\][^\n]*\n(.*?)(?=^\[P\d+\]|^Question:)", re.S | re.M)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
REFUSAL = "I can't find that in the service documentation."
MIN_SCORE = 2.0


def answer(user_message: str) -> str:
    q_match = re.search(r"^Question:\s*(.+)$", user_message, re.M)
    if not q_match:
        return REFUSAL
    q_tokens = set(tokens(q_match.group(1)))
    best: tuple[tuple, str, str] | None = None
    for order, (label, body) in enumerate(PASSAGE.findall(user_message)):
        for sentence in SENTENCE.split(body.strip()):
            s_tokens = set(tokens(sentence))
            score = sum(_weight(t) for t in q_tokens & s_tokens)
            if any(any(c.isdigit() for c in t) for t in s_tokens - q_tokens):
                score += 0.5
            key = (score, -len(s_tokens), -order)
            if best is None or key > best[0]:
                best = (key, label, sentence.strip())
    if best is None or best[0][0] < MIN_SCORE:
        return REFUSAL
    return f"{best[2]} [{best[1]}]"


def _weight(tok: str) -> float:
    if is_identifier(tok):
        return 3.0
    return 2.0 if any(c.isdigit() for c in tok) else 1.0
