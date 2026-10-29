"""fieldassist: the commands the pipelines' container tasks run, and the ones an operator runs by hand."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import corpus, evaluation
from .answer import KnownLines, answer_question
from .db import apply_schema, make_pool
from .gate import GateConfig, compare, format_table
from .ingest import ingest_dir
from .model_client import ModelClient
from .promote import PromotionError, promote
from .settings import RetrievalConfig, Settings

DEFAULT_IMAGE = ("oci://quay.example.com/fieldassist/granite-3.3-8b-instruct@sha256:"
                 "4f1c0d9a7e2b5c8d3a6f9e1b4c7d0a3f6e9b2c5d8a1f4e7b0c3d6a9f2e5b8c1d")


def _clients(s: Settings) -> tuple[ModelClient, ModelClient]:
    return (ModelClient(s.chat_url, s.chat_model, s.model_token),
            ModelClient(s.embed_url, s.embed_model, s.model_token))


def cmd_seed(args, s: Settings) -> int:
    stats = corpus.write(Path(args.data_dir or s.data_dir))
    print(json.dumps(stats))
    return 0


def cmd_ingest(args, s: Settings) -> int:
    pool = make_pool(s.database_url)
    apply_schema(pool)
    _, embedder = _clients(s)
    stats = ingest_dir(pool, embedder, Path(args.corpus or s.data_dir / "corpus"))
    print(json.dumps(stats))
    return 0


def cmd_ask(args, s: Settings) -> int:
    pool = make_pool(s.database_url)
    chat, embedder = _clients(s)
    cfg = RetrievalConfig.load(args.config or s.retrieval_config)
    ans = answer_question(pool, chat, embedder, cfg, args.question, KnownLines(pool), args.product_line)
    print(ans.text)
    if ans.refused:
        print(f"  (refused: {ans.reason})")
    for c in ans.citations:
        print(f"  [{c['label']}] {c['title']}, revision {c['revision']} · {c['section']}")
    return 0


def cmd_eval(args, s: Settings) -> int:
    pool = make_pool(s.database_url)
    if args.chat_url:
        s = replace(s, chat_url=args.chat_url)
    chat, embedder = _clients(s)
    cfg = RetrievalConfig.load(args.config)
    golden_path = Path(args.golden or s.data_dir / "golden_set.jsonl")
    report = evaluation.run(pool, chat, embedder, cfg, corpus.load_golden_set(golden_path), golden_path,
                            args.model_image)
    evaluation.write_report(report, Path(args.out))
    o = report["metrics"]["overall"]
    print(f"{report['run_id']} config={cfg.name} questions={o['questions']} accuracy={o['accuracy']} "
          f"recall@{cfg.top_k}={o['recall_at_k']} citations_active={o['citations_active']} "
          f"false_refusals={o['false_refusals']}")
    for name, m in report["metrics"]["slices"].items():
        print(f"  {name:<13} n={m['questions']:<3} accuracy={m['accuracy']:<5} recall@{cfg.top_k}={m['recall_at_k']}")
    return 0


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_gate(args, s: Settings) -> int:
    result = compare(_load(args.baseline), _load(args.candidate), GateConfig.load(args.gate_config))
    print(format_table(result))
    if args.decision_out:
        Path(args.decision_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.decision_out).write_text(result.decision, encoding="utf-8")
    return 0 if result.decision == "pass" or args.no_fail else 1


def cmd_baseline(args, s: Settings) -> int:
    """Write the report of the run currently promoted to an environment, for the pipeline's gate task."""
    pool = make_pool(s.database_url)
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT r.run_id, r.config, r.model_image, r.golden_hash, r.metrics, r.created_at FROM promotions p "
            "JOIN eval_runs r USING (run_id) WHERE p.environment = %s AND p.decision = 'pass' "
            "ORDER BY p.created_at DESC, p.id DESC LIMIT 1", (args.environment,)).fetchone()
    if row is None:
        print(f"no passing promotion recorded for {args.environment}", file=sys.stderr)
        return 1
    report = {**row, "created_at": row["created_at"].isoformat(timespec="seconds"), "failures": []}
    evaluation.write_report(report, Path(args.out))
    print(f"baseline for {args.environment}: {row['run_id']}")
    return 0


def cmd_promote(args, s: Settings) -> int:
    pool = make_pool(s.database_url) if not args.no_record else None
    try:
        p = promote(_load(args.candidate), _load(args.baseline), Path(args.overlay),
                    GateConfig.load(args.gate_config), args.environment, pool)
    except PromotionError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(format_table(p.gate))
    if p.gate.decision != "pass":
        return 1
    print(p.diff or "overlay already matches the candidate; nothing to change")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fieldassist")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("seed", help="write the synthetic corpus and golden set")
    p.add_argument("--data-dir")
    p.set_defaults(fn=cmd_seed)

    p = sub.add_parser("ingest", help="parse, chunk, embed and index the corpus; recompute active text")
    p.add_argument("--corpus")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="answer one question with citations")
    p.add_argument("question")
    p.add_argument("--config")
    p.add_argument("--product-line")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("eval", help="run the golden set against one serving configuration")
    p.add_argument("--config", required=True)
    p.add_argument("--model-image", default=DEFAULT_IMAGE)
    p.add_argument("--chat-url", help="evaluate a candidate deployment instead of FA_CHAT_URL")
    p.add_argument("--golden")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("gate", help="compare a candidate report with the production baseline")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--gate-config", default="config/gate.yaml")
    p.add_argument("--decision-out")
    p.add_argument("--no-fail", action="store_true", help="exit 0 on block (the pipeline branches on the output)")
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("baseline", help="write the report of the run currently promoted to an environment")
    p.add_argument("--environment", default="prod")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("promote", help="gate, then write the candidate into the prod overlay")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--overlay", default="deploy/overlays/prod")
    p.add_argument("--environment", default="prod")
    p.add_argument("--gate-config", default="config/gate.yaml")
    p.add_argument("--no-record", action="store_true")
    p.set_defaults(fn=cmd_promote)

    args = ap.parse_args(argv)
    return args.fn(args, Settings.from_env())


if __name__ == "__main__":
    sys.exit(main())
