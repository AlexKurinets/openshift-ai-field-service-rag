import copy
import json
import shutil

import pytest
import yaml
from conftest import ROOT, TEST_DB

from fieldassist import cli
from fieldassist.gate import GateConfig
from fieldassist.promote import MODEL_PATCH, RETRIEVAL_FILE, PromotionError, promote

GATE = GateConfig.load(ROOT / "config" / "gate.yaml")
NEW_IMAGE = "oci://quay.example.com/fieldassist/granite-3.3-8b-instruct@sha256:" + "c" * 64


@pytest.fixture
def overlay(tmp_path):
    dst = tmp_path / "prod"
    shutil.copytree(ROOT / "deploy" / "overlays" / "prod", dst)
    return dst


def test_passing_candidate_is_written_into_the_overlay_and_recorded(pool, reports, overlay):
    candidate = copy.deepcopy(reports["production"])
    candidate["model_image"] = NEW_IMAGE
    p = promote(candidate, reports["production"], overlay, GATE, "prod", pool)
    assert p.gate.decision == "pass"
    patch = yaml.safe_load((overlay / MODEL_PATCH).read_text())
    assert patch["spec"]["predictor"]["model"]["storageUri"] == NEW_IMAGE
    assert patch["metadata"]["annotations"]["fieldassist.example.com/eval-run"] == candidate["run_id"]
    assert yaml.safe_load((overlay / RETRIEVAL_FILE).read_text()) == candidate["config"]
    assert f"+      storageUri: {NEW_IMAGE}" in p.diff
    with pool.connection() as conn:
        row = conn.execute("SELECT decision, model_image FROM promotions WHERE run_id = %s",
                           (candidate["run_id"],)).fetchone()
    assert row == {"decision": "pass", "model_image": NEW_IMAGE}


def test_blocked_candidate_leaves_the_overlay_untouched(pool, reports, overlay):
    before = {p.name: p.read_text() for p in overlay.iterdir()}
    candidate = copy.deepcopy(reports["vector-only"])
    p = promote(candidate, reports["production"], overlay, GATE, "prod", pool)
    assert p.gate.decision == "block" and p.diff == ""
    assert {q.name: q.read_text() for q in overlay.iterdir()} == before


def test_model_image_must_be_pinned_by_digest(reports, overlay):
    candidate = copy.deepcopy(reports["production"])
    candidate["model_image"] = "oci://quay.example.com/fieldassist/granite-3.3-8b-instruct:latest"
    with pytest.raises(PromotionError, match="pinned by sha256"):
        promote(candidate, reports["production"], overlay, GATE, "prod")


def test_baseline_command_returns_the_run_promoted_last(pool, reports, tmp_path, monkeypatch):
    first = copy.deepcopy(reports["production"])
    promote(first, reports["production"], tmp_path, GATE, "staging", pool)
    blocked = copy.deepcopy(reports["no-revision-filter"])
    promote(blocked, reports["production"], tmp_path, GATE, "staging", pool)
    monkeypatch.setenv("FA_DATABASE_URL", TEST_DB)
    out = tmp_path / "baseline.json"
    assert cli.main(["baseline", "--environment", "staging", "--out", str(out)]) == 0
    baseline = json.loads(out.read_text())
    assert baseline["run_id"] == first["run_id"]
    assert baseline["metrics"] == first["metrics"] and baseline["golden_hash"] == first["golden_hash"]


def test_committed_prod_overlay_was_written_by_promote():
    overlay = ROOT / "deploy" / "overlays" / "prod"
    header = (overlay / MODEL_PATCH).read_text().splitlines()[0]
    run_id = header.split(" from ")[1].split(".")[0]
    patch = yaml.safe_load((overlay / MODEL_PATCH).read_text())
    assert patch["metadata"]["annotations"]["fieldassist.example.com/eval-run"] == run_id
    assert (overlay / RETRIEVAL_FILE).read_text().splitlines()[0] == header
