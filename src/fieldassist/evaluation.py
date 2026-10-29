"""Golden-set evaluation of one serving configuration (model image + retrieval config + prompt).

A question counts as correct only if the answer contains every key fact and cites a passage from
the expected document, so a right value from the wrong source (an old manual page) still fails. An
out-of-scope question is correct only if the system refuses. Recall@k asks whether the expected
section was among the passages the model was shown at all, which separates retrieval failures
from reading failures.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import uuid
from datetime import UTC, datetime
from pathlib import Path

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .answer import KnownLines, answer_question
from .model_client import ModelClient
from .settings import RetrievalConfig


def golden_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def score_question(q: dict, ans) -> dict:
    expected = (q["expected_doc"], q["expected_section"])
    retrieved = [(h.doc_id, h.section_id) for h in ans.passages]
    if q["answerable"]:
        facts_ok = all(f.lower() in ans.text.lower() for f in q["key_facts"])
        cited_ok = any(c["doc_id"] == q["expected_doc"] for c in ans.citations)
        correct = (not ans.refused) and facts_ok and cited_ok
        recall = expected in retrieved
    else:
        correct, recall = ans.refused, None
    return {
        "id": q["id"], "slice": q["slice"], "correct": correct, "recall_hit": recall, "refused": ans.refused,
        "reason": ans.reason, "citations_active": all(c["active"] for c in ans.citations),
        "latency_ms": round(ans.timings.get("total", 0.0) * 1000, 1), "answer": ans.text,
    }


def _rate(xs: list[bool]) -> float:
    return round(100.0 * sum(xs) / len(xs), 1) if xs else 0.0


def summarize(rows: list[dict]) -> dict:
    def block(rs: list[dict]) -> dict:
        answerable = [r for r in rs if r["recall_hit"] is not None]
        answered = [r for r in rs if not r["refused"]]
        return {
            "questions": len(rs),
            "accuracy": _rate([r["correct"] for r in rs]),
            "recall_at_k": _rate([r["recall_hit"] for r in answerable]) if answerable else None,
            "citations_active": _rate([r["citations_active"] for r in answered]) if answered else 100.0,
        }

    lat = sorted(r["latency_ms"] for r in rows)
    overall = block(rows)
    answerable = [r for r in rows if r["recall_hit"] is not None]
    overall["false_refusals"] = _rate([r["refused"] for r in answerable])
    overall["latency_ms_p50"] = round(statistics.median(lat), 1)
    overall["latency_ms_p95"] = round(lat[int(0.95 * (len(lat) - 1))], 1)
    slices = sorted({r["slice"] for r in rows})
    return {"overall": overall, "slices": {s: block([r for r in rows if r["slice"] == s]) for s in slices}}


def run(pool: ConnectionPool, chat: ModelClient, embedder: ModelClient, cfg: RetrievalConfig,
        golden: list[dict], golden_path: Path, model_image: str) -> dict:
    known = KnownLines(pool)
    rows = [score_question(q, answer_question(pool, chat, embedder, cfg, q["question"], known)) for q in golden]
    report = {
        "run_id": f"eval-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        "config": cfg.as_dict(),
        "model_image": model_image,
        "golden_hash": golden_hash(golden_path),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": summarize(rows),
        "failures": [r for r in rows if not r["correct"]][:40],
    }
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO eval_runs (run_id, config_name, config, model_image, golden_hash, metrics) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (run_id) DO NOTHING",
            (report["run_id"], cfg.name, Jsonb(report["config"]), model_image, report["golden_hash"],
             Jsonb(report["metrics"])))
    return report


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
