"""Policy checks on the rendered Kustomize overlays: what has to be true of anything Argo CD syncs."""

import os
import shutil
import subprocess

import pytest
import yaml
from conftest import ROOT

from fieldassist.promote import PINNED_OCI
from fieldassist.settings import RetrievalConfig

DEPLOY = ROOT / "deploy"
GPU_NODES, GPUS_PER_NODE = 2, 4


def _kustomize_cmd():
    if shutil.which("kustomize"):
        return ["kustomize", "build"]
    if shutil.which("kubectl"):
        return ["kubectl", "kustomize"]
    if os.environ.get("REQUIRE_KUSTOMIZE"):
        pytest.fail("kustomize is required in CI")
    pytest.skip("neither kustomize nor kubectl on PATH")


def render(path: str) -> list[dict]:
    out = subprocess.run([*_kustomize_cmd(), str(DEPLOY / path)], capture_output=True, text=True, check=True)
    return [d for d in yaml.safe_load_all(out.stdout) if d]


@pytest.fixture(scope="module")
def overlays():
    return {"prod": render("overlays/prod"), "dev": render("overlays/dev"), "cluster": render("cluster")}


def of_kind(docs, kind):
    return [d for d in docs if d["kind"] == kind]


def test_overlays_render_into_their_namespaces(overlays):
    for env in ("prod", "dev"):
        namespaced = [d for d in overlays[env] if d["kind"] != "Namespace"]
        assert {d["metadata"]["namespace"] for d in namespaced} == {f"fieldassist-{env}"}
        assert len(of_kind(overlays[env], "InferenceService")) == 2


def test_inference_services_are_raw_authenticated_pinned_and_never_scale_to_zero(overlays):
    for env in ("prod", "dev"):
        for isvc in of_kind(overlays[env], "InferenceService"):
            ann = isvc["metadata"]["annotations"]
            pred = isvc["spec"]["predictor"]
            assert ann["serving.kserve.io/deploymentMode"] == "RawDeployment"
            assert ann["security.opendatahub.io/enable-auth"] == "true"
            assert pred["minReplicas"] >= 1
            assert PINNED_OCI.match(pred["model"]["storageUri"]), pred["model"]["storageUri"]
            assert pred["model"]["resources"]["limits"]["nvidia.com/gpu"] == "1"
            assert {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"} in pred["tolerations"]


def test_chat_model_serving_flags(overlays):
    (chat,) = [i for i in of_kind(overlays["prod"], "InferenceService") if i["metadata"]["name"] == "granite-chat"]
    args = chat["spec"]["predictor"]["model"]["args"]
    assert "--max-model-len=8192" in args and "--gpu-memory-utilization=0.90" in args


def test_prod_chat_rollout_never_needs_a_gpu_it_does_not_have(overlays):
    (chat,) = [i for i in of_kind(overlays["prod"], "InferenceService") if i["metadata"]["name"] == "granite-chat"]
    pred = chat["spec"]["predictor"]
    assert pred["minReplicas"] == pred["maxReplicas"] == 2
    assert pred["deploymentStrategy"]["rollingUpdate"] == {"maxSurge": 0, "maxUnavailable": 1}


def test_gpu_budget_fits_two_nodes_of_four_l40s(overlays):
    serving, surge = 0, 0
    for env in ("prod", "dev"):
        for isvc in of_kind(overlays[env], "InferenceService"):
            pred = isvc["spec"]["predictor"]
            gpus = int(pred["model"]["resources"]["limits"]["nvidia.com/gpu"])
            serving += pred["maxReplicas"] * gpus
            surge = max(surge, int(pred["deploymentStrategy"]["rollingUpdate"]["maxSurge"]) * gpus)
    (cq,) = of_kind(overlays["cluster"], "ClusterQueue")
    resources = cq["spec"]["resourceGroups"][0]["flavors"][0]["resources"]
    kueue = int(next(r["nominalQuota"] for r in resources if r["name"] == "nvidia.com/gpu"))
    assert (serving, surge, kueue) == (5, 1, 2)
    assert serving + surge + kueue <= GPU_NODES * GPUS_PER_NODE


def test_prod_serves_hybrid_retrieval_with_the_revision_filter(overlays):
    (cm,) = [c for c in of_kind(overlays["prod"], "ConfigMap")
             if c["metadata"]["name"].startswith("fieldassist-retrieval")]
    raw = yaml.safe_load(cm["data"]["retrieval.yaml"])
    cfg = RetrievalConfig(**raw)
    assert cfg.mode == "hybrid" and cfg.active_only
    (deploy,) = of_kind(overlays["prod"], "Deployment")
    assert deploy["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] == cm["metadata"]["name"]


def test_every_container_sets_requests_and_limits(overlays):
    for env in ("prod", "dev"):
        for d in of_kind(overlays[env], "Deployment"):
            for c in d["spec"]["template"]["spec"]["containers"]:
                assert c["resources"]["requests"] and c["resources"]["limits"], c["name"]
        for isvc in of_kind(overlays[env], "InferenceService"):
            res = isvc["spec"]["predictor"]["model"]["resources"]
            assert res["requests"] and res["limits"]


def test_images_are_pinned_in_prod(overlays):
    (deploy,) = of_kind(overlays["prod"], "Deployment")
    assert "@sha256:" in deploy["spec"]["template"]["spec"]["containers"][0]["image"]
    (runtime,) = of_kind(overlays["prod"], "ServingRuntime")
    assert "@sha256:" in runtime["spec"]["containers"][0]["image"]


def test_argocd_applications_point_at_paths_that_exist():
    apps = list(yaml.safe_load_all((DEPLOY / "argocd" / "applications.yaml").read_text()))
    assert {a["metadata"]["name"] for a in apps} == {"fieldassist-cluster", "fieldassist-dev", "fieldassist-prod"}
    for a in apps:
        assert (ROOT / a["spec"]["source"]["path"] / "kustomization.yaml").exists()
        assert a["spec"]["syncPolicy"]["automated"]["selfHeal"] is True
