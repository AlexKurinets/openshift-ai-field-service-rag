"""Promotion: write the gated configuration into the prod overlay that Argo CD syncs.

In the cluster the last task of the eval pipeline runs this and opens a pull request with the
result; a person merges it and Argo CD rolls it out. Here the Git host is abstracted away: the
overlay files are rewritten in place and the diff is printed, which is what the pull request
would contain. Nothing reaches prod without passing the gate, and a model image must be pinned by
digest so the image that was evaluated is the image that runs.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .gate import GateConfig, GateResult, compare

PINNED_OCI = re.compile(r"^oci://[a-z0-9.\-:/]+@sha256:[0-9a-f]{64}$")
MODEL_PATCH = "granite-chat-model.yaml"
RETRIEVAL_FILE = "retrieval.yaml"


class PromotionError(RuntimeError):
    pass


@dataclass
class Promotion:
    gate: GateResult
    diff: str


def _dump(obj: dict) -> str:
    return yaml.safe_dump(obj, sort_keys=False)


def render_overlay_files(candidate: dict) -> dict[str, str]:
    patch = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {"name": "granite-chat", "annotations": {"fieldassist.example.com/eval-run": candidate["run_id"]}},
        "spec": {"predictor": {"model": {"storageUri": candidate["model_image"]}}},
    }
    retrieval = {k: v for k, v in candidate["config"].items()}
    header = f"# Written by `fieldassist promote` from {candidate['run_id']}. Do not edit by hand.\n"
    return {MODEL_PATCH: header + _dump(patch), RETRIEVAL_FILE: header + _dump(retrieval)}


def promote(candidate: dict, baseline: dict, overlay: Path, gate_cfg: GateConfig, environment: str,
            pool: ConnectionPool | None = None) -> Promotion:
    if not PINNED_OCI.match(candidate["model_image"]):
        raise PromotionError(f"model image must be an oci:// reference pinned by sha256 digest, "
                             f"got {candidate['model_image']!r}")
    result = compare(baseline, candidate, gate_cfg)
    diff = ""
    if result.decision == "pass":
        chunks = []
        for name, new in render_overlay_files(candidate).items():
            path = overlay / name
            old = path.read_text(encoding="utf-8") if path.exists() else ""
            if old != new:
                chunks.append("".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                           f"a/{path.as_posix()}", f"b/{path.as_posix()}")))
                path.write_text(new, encoding="utf-8", newline="\n")
        diff = "".join(chunks)
    if pool is not None:
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO promotions (environment, run_id, baseline_run, decision, reasons, model_image) "
                "SELECT %s, %s, %s, %s, %s, %s WHERE EXISTS (SELECT 1 FROM eval_runs WHERE run_id = %s)",
                (environment, candidate["run_id"], baseline["run_id"], result.decision, Jsonb(result.reasons),
                 candidate["model_image"], candidate["run_id"]))
    return Promotion(result, diff)
