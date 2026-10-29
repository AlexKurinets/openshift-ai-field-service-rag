"""Promotion gate for a candidate deployed in dev: a new model image, a retrieval change or a prompt
version. The candidate runs the 300-question golden set, the gate compares it slice by slice with the
report of the run currently promoted to prod, and only a pass reaches the promotion task, which
writes the prod overlay (in production it opened the pull request that a person merged)."""

from components import evaluate, fetch_baseline, gate, promote, with_runtime_env
from kfp import dsl


@dsl.pipeline(name="fieldassist-eval-gate",
              description="Golden-set evaluation of a candidate, slice-by-slice gate, promotion on pass.")
def eval_gate(candidate_model_image: str,
              candidate_config: str = "config/retrieval/production.yaml",
              candidate_chat_url: str = "http://granite-chat-predictor.fieldassist-dev.svc:8080",
              environment: str = "prod"):
    baseline = with_runtime_env(fetch_baseline(environment=environment))
    baseline.set_display_name("baseline: run serving prod")

    candidate = with_runtime_env(evaluate(config_path=candidate_config, model_image=candidate_model_image,
                                          chat_url=candidate_chat_url))
    candidate.set_display_name("golden set on candidate")

    decision = with_runtime_env(gate(baseline=baseline.outputs["report"], candidate=candidate.outputs["report"]))
    decision.set_display_name("gate")

    with dsl.If(decision.outputs["decision"] == "pass", name="gate-passed"):
        promoted = with_runtime_env(promote(baseline=baseline.outputs["report"],
                                            candidate=candidate.outputs["report"], environment=environment))
        promoted.set_display_name("write prod overlay")
