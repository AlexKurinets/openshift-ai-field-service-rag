import subprocess
import sys

import pytest
import yaml
from conftest import ROOT

pytest.importorskip("kfp")

COMPILED = ROOT / "pipelines" / "compiled"


def load(name):
    return list(yaml.safe_load_all((COMPILED / name).read_text()))


def test_committed_pipeline_yaml_matches_the_python_source():
    out = subprocess.run([sys.executable, str(ROOT / "pipelines" / "compile.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_promotion_runs_only_inside_the_gate_passed_branch():
    spec, _platform = load("eval_gate.yaml")
    root = spec["root"]["dag"]["tasks"]
    assert set(root) == {"fetch-baseline", "evaluate", "gate", "condition-1"}
    cond = root["condition-1"]
    assert "== 'pass'" in cond["triggerPolicy"]["condition"]
    assert "gate" in cond["dependentTasks"]
    inner = spec["components"]["comp-condition-1"]["dag"]["tasks"]
    assert set(inner) == {"promote"}


def test_every_task_gets_the_database_secret_and_no_cached_results():
    spec, platform = load("eval_gate.yaml")
    executors = platform["platforms"]["kubernetes"]["deploymentSpec"]["executors"]
    for name in ("exec-evaluate", "exec-fetch-baseline", "exec-gate", "exec-promote"):
        secrets = {s["secretName"] for s in executors[name]["secretAsEnv"]}
        assert secrets == {"fieldassist-db", "fieldassist-model-token"}
    for task in spec["root"]["dag"]["tasks"].values():
        if "cachingOptions" in task:
            assert task["cachingOptions"] == {} or task["cachingOptions"].get("enableCache") is False


def test_nightly_ingest_evaluates_after_ingesting():
    spec, _ = load("nightly_ingest.yaml")
    tasks = spec["root"]["dag"]["tasks"]
    assert tasks["evaluate"]["dependentTasks"] == ["ingest"]
    args = spec["deploymentSpec"]["executors"]["exec-evaluate"]["container"]["args"]
    assert "--config" in args
    assert spec["root"]["inputDefinitions"]["parameters"]["config_path"]["defaultValue"] == \
        "deploy/overlays/prod/retrieval.yaml"
