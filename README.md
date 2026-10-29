# openshift-ai-field-service-rag

A question-answering service for field-service engineers, running on Red Hat OpenShift AI. An engineer asks about a torque value, a fault code, a part number or a service interval, and the gateway retrieves the current text from service manuals and service bulletins, has Granite 3.3 8B Instruct answer from those passages only, checks that every answer cites a passage it was given, and refuses when it can't ground one. The models run on KServe with the vLLM runtime. Ingestion and evaluation run as Data Science Pipelines, and a change to the model, the prompt or retrieval reaches production only if a 300-question golden set says it doesn't regress, through a Kustomize overlay that Argo CD syncs.

This is a reference implementation of the architecture behind a platform I built for a mid-sized industrial equipment manufacturer. The production build is under NDA, so a few integrations are abstracted behind adapters: an OpenAI-compatible stand-in replaces the two vLLM model servers, documents arrive as Markdown rather than as an export from a document management system, and promotion rewrites the overlay and prints the diff instead of opening a pull request. It runs on synthetic data, and every company, product, part number, place, person and figure in it is made up.

[![ci](https://github.com/AlexKurinets/openshift-ai-field-service-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexKurinets/openshift-ai-field-service-rag/actions/workflows/ci.yml)

## Architecture

```mermaid
flowchart LR
    APP["Field-service app"] -->|"POST /v1/ask"| GW
    subgraph NS["OpenShift AI data science project: fieldassist-prod"]
        GW["fieldassist-gateway<br/>FastAPI, 2 replicas"]
        EMB["granite-embed<br/>KServe RawDeployment + vLLM<br/>768-dim, 1 x L40S"]
        CHAT["granite-chat<br/>KServe RawDeployment + vLLM<br/>Granite 3.3 8B, 2 x L40S"]
        PG[("Postgres + pgvector<br/>chunks, supersessions,<br/>eval_runs, promotions")]
        DSP["Data Science Pipelines<br/>nightly ingest, eval gate"]
    end
    GW -->|"/v1/embeddings"| EMB
    GW -->|"hybrid retrieval"| PG
    GW -->|"/v1/chat/completions"| CHAT
    DSP -->|"chunks, eval runs"| PG
    DSP -->|"300-question golden set"| CHAT
    DSP -->|"gate passed: overlay diff"| GIT["Git: deploy/overlays/prod"]
    GIT -->|"Argo CD sync"| NS
```

The cluster has two GPU worker nodes with four NVIDIA L40S each. Five GPUs serve models (two prod chat replicas, prod embeddings, dev chat, dev embeddings), one stays free so a single-replica model server can surge during a rollout, and two are shared through a Kueue ClusterQueue by workbenches, eval jobs and fine-tuning experiments.

## Design decisions

**RawDeployment, never scaled to zero, weights shipped as images.** My first version served the chat model in KServe's Serverless mode with scale-to-zero. Pulling about 16 GB of weights from object storage on a cold start took around four minutes, so the first question of every morning timed out. Both InferenceServices now run as RawDeployment (newer KServe releases call it Standard) with `minReplicas` of at least 1, and the weights ship as OCI modelcar images referenced by digest (`storageUri: oci://...@sha256:...`). Nodes cache them like any other image, and the image that passed the gate is byte for byte the image that runs. Token auth is on for both (`security.opendatahub.io/enable-auth`). See [`deploy/base/isvc-granite-chat.yaml`](deploy/base/isvc-granite-chat.yaml); [`tests/test_manifests.py`](tests/test_manifests.py) fails the build if any InferenceService drops one of these properties.

**Rollouts fit inside the GPUs that exist.** With every GPU allocated, a default rolling update would schedule a third chat replica and wait forever for a GPU. Prod chat runs exactly two replicas with `maxSurge: 0, maxUnavailable: 1` ([`deploy/overlays/prod/granite-chat-replicas.yaml`](deploy/overlays/prod/granite-chat-replicas.yaml)), so it replaces one replica at a time inside its own two GPUs while the other keeps serving. `test_gpu_budget_fits_two_nodes_of_four_l40s` adds up serving replicas, the largest surge and the Kueue quota from the rendered overlays and checks the total against eight.

**Hybrid retrieval, with exact identifiers first.** Dense embeddings put every part-number chunk in the same neighbourhood, since 4434-3059 and 4434-3725 mean nothing to them. [`retrieval.py`](src/fieldassist/retrieval.py) runs pgvector cosine search and Postgres full-text search side by side, 20 candidates each, and merges them with reciprocal rank fusion (k = 60), which works on ranks alone so the two retrievers' scores never need calibrating. Each query term is quoted inside `to_tsquery`, so an identifier goes through the same text parser that indexed it and has to match as a phrase. Pure RRF still lost fault codes: a chunk ranked first by keyword alone scores 1/61, which loses to near-identical chunks sitting lower on both lists (1/63 + 1/62). A chunk containing a part number or fault code from the question is therefore always shown to the model first.

**Which text is current is computed, never decided at insert time.** A service bulletin reissues a whole manual section with one value changed, and the manual stays in the index. Documents arrive in no guaranteed order, so after every ingestion run one `UPDATE` in [`ingest.py`](src/fieldassist/ingest.py) recomputes `chunks.active` from the latest revision of each document and the `supersessions` table. A bulletin ingested before the manual it amends still wins, and a test covers exactly that order. The HNSW index is partial on `active`, so a filtered query uses it directly instead of post-filtering a full index scan and coming back with fewer than k rows ([`schema.sql`](src/fieldassist/schema.sql)).

**Refusals are code paths, not prompt requests.** [`answer.py`](src/fieldassist/answer.py) refuses before calling the model when the question names a model line with no documentation (`KR-300`; a bulletin number such as `SB-2026-004` is not mistaken for one), and when the best cosine similarity is under 0.2 with no exact identifier match. After generation it refuses an answer that cites no passage, or cites a passage it was never given. The model is told the same rules, but the gateway doesn't rely on it following them.

**The gate compares slices, and promotion goes through Git.** A retrieval change that helps paraphrased questions can quietly break part-number lookups, and an overall average hides that. [`gate.py`](src/fieldassist/gate.py) blocks when overall accuracy, any slice's accuracy or recall@5 falls more than 2 points, when the share of citations pointing at current text falls at all, when fewer than 300 questions ran, or when the two runs used different golden sets. A question only counts as correct if the answer has every key fact and cites the expected document, so the right number taken from a superseded page still fails. The baseline is the stored report of the run currently promoted to prod (`fieldassist baseline`). On a pass, [`promote.py`](src/fieldassist/promote.py) writes the chat model's `storageUri` and the retrieval config into the prod overlay, refusing any model image that isn't pinned by digest. The retrieval config becomes a hashed ConfigMap, so a change rolls the gateway. In production that task opened the pull request, a person merged it, and Argo CD synced it.

**One image for the gateway, the pipelines and the CLI.** Every Data Science Pipelines task in [`pipelines/components.py`](pipelines/components.py) runs the same image as the gateway with a `fieldassist` subcommand, so a pipeline step and a local command are the same code. The eval-gate pipeline branches on the gate's output with `dsl.If`, and promotion exists only inside that branch. The image gives group 0 the owner's permissions because OpenShift runs containers as an arbitrary UID in group 0.

## What the golden set shows

The golden set is 300 questions in seven slices: specs 45, torque 60, intervals 45, fault codes 60, part numbers 42, values a bulletin changed 24, and out of scope 24 (unrelated questions and model lines that don't exist). These are the three configurations in [`config/retrieval`](config/retrieval) on the Compose stack:

| Configuration | Overall | Fault codes | Part numbers | Superseded values | Citations to current text | Gate |
|---|---|---|---|---|---|---|
| `production`: hybrid, revision filter | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | baseline |
| `vector-only` | 74.7 | 31.7 | 83.3 | 100.0 | 100.0 | blocked |
| `no-revision-filter` | 92.0 | 100.0 | 78.6 | 75.0 | 91.7 | blocked |

The 100.0 is not a claim about Granite. The corpus and the questions are generated from the same facts, and the model stand-in is a deterministic extractive reader, so that row shows the plumbing works. The deltas are the point of the table. Vector-only retrieval can't separate fault-code chunks and loses 68 points on that slice. Dropping the revision filter answers some superseded values from the old manual page, and it also loses unchanged values in reissued sections, because those answers cite the superseded page. The gate blocks both.

## Running it

Docker Compose brings up Postgres with pgvector, the model stand-in and the gateway. The gateway ingests the synthetic corpus on start and reports healthy once the index and both models answer. Host ports default to 5496, 8171 and 8170 and can be changed with `POSTGRES_PORT`, `MODELS_PORT` and `API_PORT`.

```bash
docker compose up -d --build --wait
curl -s localhost:8170/readyz
```

Ask questions. The first answer comes from service bulletin SB-2026-004, which replaced the manual's value, the second is found by exact fault-code match, and the third is refused because no TD-150 exists.

```bash
ask() { curl -s -H "Authorization: Bearer local-dev-token" -H "Content-Type: application/json" \
          -d "{\"question\": \"$1\"}" localhost:8170/v1/ask; echo; }
ask "What torque do the airend coupling bolts on the KR-160 take?"
ask "What does fault F6580 mean?"
ask "How often should I change the oil on the TD-150?"
curl -s -H "Authorization: Bearer local-dev-token" localhost:8170/v1/index/status
```

Run the golden set against the three configurations, then gate the two weaker ones against production. Both gate commands print a slice table, end in `decision: BLOCK` and exit 1.

```bash
docker compose exec gateway fieldassist eval --config config/retrieval/production.yaml --out reports/production.json
docker compose exec gateway fieldassist eval --config config/retrieval/vector-only.yaml --out reports/vector-only.json
docker compose exec gateway fieldassist eval --config config/retrieval/no-revision-filter.yaml --out reports/no-revision-filter.json
docker compose exec gateway fieldassist gate --baseline reports/production.json --candidate reports/vector-only.json
docker compose exec gateway fieldassist gate --baseline reports/production.json --candidate reports/no-revision-filter.json
```

Promote a new model image. The first command records production as the prod baseline, the candidate is the same configuration with a new image digest, and the last command passes the gate and prints the overlay diff that would go into the pull request.

```bash
docker compose exec gateway fieldassist promote --baseline reports/production.json --candidate reports/production.json
docker compose exec gateway fieldassist eval --config config/retrieval/production.yaml \
  --model-image oci://quay.example.com/fieldassist/granite-3.3-8b-instruct@sha256:5e2b8d1f4a7c0e3b6d9f2a5c8e1b4d7f0a3c6e9b2d5f8a1c4e7b0d3f6a9c2e5b \
  --out reports/candidate.json
docker compose exec gateway fieldassist baseline --environment prod --out reports/baseline.json
docker compose exec gateway fieldassist promote --baseline reports/baseline.json --candidate reports/candidate.json
```

[`scripts/smoke.sh`](scripts/smoke.sh) runs every block above in order with assertions, and CI runs it on every push. `docker compose down -v` removes the stack and its data.

Without Docker you need Python 3.11 or later and a Postgres with the pgvector extension:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,pipelines]"
export FA_DATABASE_URL=postgresql://fieldassist:fieldassist@localhost:5496/fieldassist
uvicorn model_sim.app:app --port 8171 &
fieldassist seed      # regenerates data/ from the fixed seed; identical to the committed files
fieldassist ingest
fieldassist ask "What is part 4434-3059?"
```

On a cluster, `deploy/argocd/applications.yaml` points Argo CD at `deploy/cluster`, `deploy/overlays/dev` and `deploy/overlays/prod`. The secrets the overlays reference (`fieldassist-db`, `fieldassist-api`, `fieldassist-model-token`, `dspa-s3`) are created out of band, and `python pipelines/compile.py` writes the IR YAML to upload to Data Science Pipelines.

## Tests

```bash
docker compose up -d postgres
pytest
```

The 57 tests run against real Postgres with pgvector, using the database in `TEST_DATABASE_URL` (default `fieldassist_test` on port 5496, created if missing), with the model stand-in served in-process. They cover the seed's determinism and the golden set's facts, ingestion order and revision handling, RRF and the exact-identifier rule, each refusal path, retries on 503 during rollouts, the gate's thresholds, promotion and the baseline lookup, the API, policy checks on the rendered overlays (they need `kustomize` or `kubectl`), and the compiled pipelines. CI also validates every rendered manifest with kubeconform in strict mode against the CRD schemas for KServe, Kueue, Argo CD and the Prometheus operator. Route and DataSciencePipelinesApplication have no published schema, so they are skipped by name.

## Layout

```
config/retrieval/          serving configurations the gate evaluates; gate.yaml holds the thresholds
data/                      synthetic corpus (6 manuals, 8 bulletins) and golden_set.jsonl, output of `fieldassist seed`
deploy/base/               ServingRuntime, InferenceServices, gateway, Route, NetworkPolicy, DSPA, LocalQueue, alerts
deploy/overlays/{dev,prod} namespaces, model images, replicas, retrieval config (prod files written by promote)
deploy/cluster/            Kueue ResourceFlavor and ClusterQueue for the shared GPUs
deploy/argocd/             Argo CD applications
pipelines/                 KFP v2 pipelines (nightly ingest, eval gate) and their compiled YAML
src/fieldassist/           gateway, ingestion, retrieval, answering, evaluation, gate, promotion, CLI
src/model_sim/             OpenAI-compatible stand-in for the two vLLM InferenceServices
tests/                     pytest suite
scripts/smoke.sh           the README, end to end, against the Compose stack
```

## What is deliberately left out

- The real models. The stand-in speaks vLLM's OpenAI-compatible API with the same bearer-token check, and `SIM_FAIL_FIRST` makes it answer 503 the way a KServe route does mid-rollout, but it is an extractive reader, so latency and accuracy figures from it say nothing about Granite on an L40S.
- Document parsing. Production parsed PDF exports with tables and scanned pages. Here the documents are Markdown with front matter, and parsing is one function.
- The Git host. Promotion rewrites the overlay and prints the diff; opening the pull request is not reproduced.
- Secret management, the OAuth proxy in front of the gateway and the field-service app itself.
- Workbench images and fine-tuning jobs. The Kueue queue they use is defined, but nothing here submits to it.

## License

MIT, Alex Kurinets.
