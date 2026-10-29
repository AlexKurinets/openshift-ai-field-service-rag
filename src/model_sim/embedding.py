"""Stand-in for the Granite embedding model: 768-dim hashed features, L2-normalized.

Words and character trigrams are hashed into the vector, so paraphrases that share vocabulary land
close together. Identifiers with four or more digits (part numbers, fault codes) are reduced to
their shape before hashing, so 4471-0213 and 4471-0388 embed identically. That reproduces the
failure we saw with the real dense model: it put all part-number and fault-code chunks in the same
neighborhood and could not tell them apart, which is why retrieval is hybrid.
"""

from __future__ import annotations

import hashlib
import math

from .text import is_identifier, shape, tokens

DIM = 768


def _slot(feature: str) -> tuple[int, float]:
    h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(h, "little")
    return n % DIM, (1.0 if (n >> 40) & 1 else -1.0)


def embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    counts: dict[str, float] = {}
    for tok in tokens(text):
        if is_identifier(tok):
            counts["w:" + shape(tok)] = counts.get("w:" + shape(tok), 0.0) + 1.0
            continue
        counts["w:" + tok] = counts.get("w:" + tok, 0.0) + 1.0
        if tok.isalpha() and len(tok) > 3:
            padded = f"^{tok}$"
            for i in range(len(padded) - 2):
                key = "c:" + padded[i:i + 3]
                counts[key] = counts.get(key, 0.0) + 0.3
    for feature, tf in counts.items():
        i, sign = _slot(feature)
        vec[i] += sign * (1.0 + math.log(tf)) if tf >= 1 else sign * tf
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
