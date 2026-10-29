"""Gateway API that the field-service app calls. Runs as the fieldassist-gateway Deployment."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .answer import KnownLines, answer_question
from .db import apply_schema, make_pool
from .model_client import ModelClient
from .settings import RetrievalConfig, Settings

ASK_SECONDS = Histogram("fieldassist_ask_seconds", "End-to-end /v1/ask latency", ["outcome"],
                        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16))
REFUSALS = Counter("fieldassist_refusals_total", "Refused questions by reason", ["reason"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    product_line: str | None = None


def create_app(settings: Settings | None = None, http: httpx.Client | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = make_pool(settings.database_url)
        apply_schema(pool)
        app.state.pool = pool
        app.state.cfg = RetrievalConfig.load(settings.retrieval_config)
        app.state.chat = ModelClient(settings.chat_url, settings.chat_model, settings.model_token, http)
        app.state.embedder = ModelClient(settings.embed_url, settings.embed_model, settings.model_token, http)
        app.state.known = KnownLines(pool)
        yield
        pool.close()

    app = FastAPI(title="fieldassist gateway", lifespan=lifespan)

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if authorization != f"Bearer {settings.api_token}":
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    @app.post("/v1/ask", dependencies=[Depends(require_token)])
    def ask(req: AskRequest):
        s = app.state
        t0 = time.monotonic()
        ans = answer_question(s.pool, s.chat, s.embedder, s.cfg, req.question, s.known, req.product_line)
        ASK_SECONDS.labels("refused" if ans.refused else "answered").observe(time.monotonic() - t0)
        if ans.refused:
            REFUSALS.labels(ans.reason.split(" (")[0] if ans.reason else "unknown").inc()
        return {
            "answer": ans.text, "refused": ans.refused, "reason": ans.reason, "citations": ans.citations,
            "config": s.cfg.name, "timings_ms": {k: round(v * 1000, 1) for k, v in ans.timings.items()},
        }

    @app.get("/v1/index/status", dependencies=[Depends(require_token)])
    def index_status():
        with app.state.pool.connection() as conn:
            docs = conn.execute("SELECT kind, count(*) AS n FROM documents GROUP BY kind ORDER BY kind").fetchall()
            counts = conn.execute("SELECT count(*) FILTER (WHERE active) AS active, count(*) AS total "
                                  "FROM chunks").fetchone()
            superseded = conn.execute(
                "SELECT doc_id, section_id, superseded_by, count(*) AS chunks FROM chunks WHERE NOT active "
                "GROUP BY doc_id, section_id, superseded_by ORDER BY doc_id, section_id").fetchall()
        return {"documents": {r["kind"]: r["n"] for r in docs}, "chunks": counts,
                "superseded_sections": superseded, "retrieval_config": app.state.cfg.as_dict()}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(response: Response):
        s = app.state
        with s.pool.connection() as conn:
            chunks = conn.execute("SELECT count(*) AS n FROM chunks WHERE active").fetchone()["n"]
        checks = {"index": chunks > 0, "chat_model": s.chat.ready(), "embedding_model": s.embedder.ready()}
        if not all(checks.values()):
            response.status_code = 503
        return {"ready": all(checks.values()), "checks": checks, "active_chunks": chunks}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
