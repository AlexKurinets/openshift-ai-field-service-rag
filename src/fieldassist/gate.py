"""Promotion gate: a candidate may not regress against the configuration serving production.

Compared slice by slice, because an overall average hides the failure that matters: a retrieval
change that helps paraphrased questions can quietly break part-number and fault-code lookups, which
are about a third of the set. Any slice, or the overall score, dropping more than max_regression_points
(percentage points) blocks promotion, as does any drop in the share of citations pointing at
current text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GateConfig:
    max_regression_points: float = 2.0
    min_questions: int = 300
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "recall_at_k"])

    @classmethod
    def load(cls, path: Path | str) -> GateConfig:
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class GateResult:
    decision: str  # pass | block
    reasons: list[str]
    rows: list[dict]


def compare(baseline: dict, candidate: dict, cfg: GateConfig) -> GateResult:
    reasons: list[str] = []
    rows: list[dict] = []
    if baseline["golden_hash"] != candidate["golden_hash"]:
        reasons.append("golden sets differ; runs are not comparable")
    n = candidate["metrics"]["overall"]["questions"]
    if n < cfg.min_questions:
        reasons.append(f"only {n} questions evaluated; the gate needs {cfg.min_questions}")

    scopes = [("overall", baseline["metrics"]["overall"], candidate["metrics"]["overall"])]
    for name, b in baseline["metrics"]["slices"].items():
        c = candidate["metrics"]["slices"].get(name)
        if c is None:
            reasons.append(f"slice {name} missing from candidate")
            continue
        scopes.append((name, b, c))
    for name, b, c in scopes:
        for metric in cfg.metrics:
            if b.get(metric) is None or c.get(metric) is None:
                continue
            delta = round(c[metric] - b[metric], 1)
            failed = delta < -cfg.max_regression_points
            rows.append({"scope": name, "metric": metric, "baseline": b[metric], "candidate": c[metric],
                         "delta": delta, "ok": not failed})
            if failed:
                reasons.append(f"{name} {metric} fell {abs(delta):.1f} points ({b[metric]} -> {c[metric]})")
    bo, co = baseline["metrics"]["overall"], candidate["metrics"]["overall"]
    if co["citations_active"] < bo["citations_active"]:
        reasons.append(f"citations to current text fell ({bo['citations_active']} -> {co['citations_active']})")
        rows.append({"scope": "overall", "metric": "citations_active", "baseline": bo["citations_active"],
                     "candidate": co["citations_active"],
                     "delta": round(co["citations_active"] - bo["citations_active"], 1), "ok": False})
    return GateResult("block" if reasons else "pass", reasons, rows)


def format_table(result: GateResult) -> str:
    lines = [f"{'scope':<14}{'metric':<18}{'baseline':>9}{'candidate':>11}{'delta':>8}  "]
    for r in result.rows:
        lines.append(f"{r['scope']:<14}{r['metric']:<18}{r['baseline']:>9}{r['candidate']:>11}"
                     f"{r['delta']:>+8.1f}  {'ok' if r['ok'] else 'REGRESSION'}")
    lines.append(f"decision: {result.decision.upper()}")
    lines += [f"  - {x}" for x in result.reasons]
    return "\n".join(lines)
