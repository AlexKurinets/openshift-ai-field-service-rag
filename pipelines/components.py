"""Container components shared by both pipelines. Every task runs the same fieldassist image the
gateway runs, so a pipeline step and a local `fieldassist ...` command are the same code path.

No `from __future__ import annotations` here: KFP reads the parameter annotations at decoration time."""

import os

from kfp import dsl, kubernetes

IMAGE = os.environ.get(
    "FA_PIPELINE_IMAGE", "image-registry.openshift-image-registry.svc:5000/fieldassist-dev/fieldassist:main")


@dsl.container_component
def ingest(corpus_dir: str, stats: dsl.Output[dsl.Artifact]):
    return dsl.ContainerSpec(
        image=IMAGE, command=["sh", "-c"],
        args=['fieldassist ingest --corpus "$0" | tee "$1"', corpus_dir, stats.path])


@dsl.container_component
def evaluate(config_path: str, model_image: str, chat_url: str, report: dsl.Output[dsl.Artifact]):
    return dsl.ContainerSpec(
        image=IMAGE, command=["fieldassist", "eval"],
        args=["--config", config_path, "--model-image", model_image, "--chat-url", chat_url, "--out", report.path])


@dsl.container_component
def fetch_baseline(environment: str, report: dsl.Output[dsl.Artifact]):
    return dsl.ContainerSpec(
        image=IMAGE, command=["fieldassist", "baseline"],
        args=["--environment", environment, "--out", report.path])


@dsl.container_component
def gate(baseline: dsl.Input[dsl.Artifact], candidate: dsl.Input[dsl.Artifact], decision: dsl.OutputPath(str)):
    return dsl.ContainerSpec(
        image=IMAGE, command=["fieldassist", "gate"],
        args=["--baseline", baseline.path, "--candidate", candidate.path, "--decision-out", decision,
              "--no-fail"])


@dsl.container_component
def promote(baseline: dsl.Input[dsl.Artifact], candidate: dsl.Input[dsl.Artifact], environment: str,
            diff: dsl.Output[dsl.Artifact]):
    return dsl.ContainerSpec(
        image=IMAGE, command=["sh", "-c"],
        args=['fieldassist promote --baseline "$0" --candidate "$1" --environment "$2" | tee "$3"',
              baseline.path, candidate.path, environment, diff.path])


def with_runtime_env(task: dsl.PipelineTask) -> dsl.PipelineTask:
    """Database URL and model token from secrets; model endpoints default to this project's
    InferenceServices (the eval task can point the chat model elsewhere with --chat-url)."""
    kubernetes.use_secret_as_env(task, secret_name="fieldassist-db", secret_key_to_env={"url": "FA_DATABASE_URL"})
    kubernetes.use_secret_as_env(task, secret_name="fieldassist-model-token",
                                 secret_key_to_env={"token": "FA_MODEL_TOKEN"})
    task.set_env_variable("FA_CHAT_URL", "http://granite-chat-predictor:8080")
    task.set_env_variable("FA_EMBED_URL", "http://granite-embed-predictor:8080")
    task.set_caching_options(False)
    return task
