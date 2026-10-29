"""Nightly recurring run: ingest the document export, then run the golden set against prod's
configuration so a bad ingestion (a parser change, a bulletin that superseded the wrong section)
shows up in the morning's report instead of in an engineer's answer. The image is built from main, so
deploy/overlays/prod/retrieval.yaml inside it is the configuration prod is serving."""

from components import evaluate, ingest, with_runtime_env
from kfp import dsl


@dsl.pipeline(name="fieldassist-nightly-ingest",
              description="Parse, chunk, embed and index service documents; then evaluate prod's configuration.")
def nightly_ingest(corpus_dir: str = "data/corpus",
                   config_path: str = "deploy/overlays/prod/retrieval.yaml",
                   model_image: str = "deployed-prod"):
    ingested = with_runtime_env(ingest(corpus_dir=corpus_dir))
    ingested.set_display_name("ingest documents")
    report = with_runtime_env(evaluate(config_path=config_path, model_image=model_image,
                                       chat_url="http://granite-chat-predictor:8080"))
    report.set_display_name("golden set on prod config")
    report.after(ingested)
