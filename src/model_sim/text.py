from __future__ import annotations

import re

TOKEN = re.compile(r"[a-z0-9]+(?:[-.,/][a-z0-9]+)*")
STOPWORDS = frozenset("""
a an and are as at be by can do does for from give gives go goes how i in is it its me my of on or our
should so that the their them then there these they this to up us was we what when where which who why
will with you your after before many much any belong belongs mean means
""".split())


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def is_identifier(tok: str) -> bool:
    """Part numbers and fault codes: four or more digits."""
    return sum(c.isdigit() for c in tok) >= 4


def shape(tok: str) -> str:
    return re.sub(r"[0-9]", "d", tok)
