"""OpenAI-compatible stand-in for the two vLLM InferenceServices (chat and embeddings).

Same request and response shapes as vLLM's OpenAI server, bearer-token check included, so the
gateway's ModelClient runs unchanged against this and against KServe. SIM_FAIL_FIRST makes the
first N requests answer 503, which is how a KServe route behaves while replicas roll.
"""

from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import embedding, extractive

CHAT_MODEL = os.environ.get("SIM_CHAT_MODEL", "granite-chat")
EMBED_MODEL = os.environ.get("SIM_EMBED_MODEL", "granite-embed")
TOKEN = os.environ.get("SIM_TOKEN", "local-model-token")

app = FastAPI(title="vLLM OpenAI-compatible stand-in")
_state = {"fail_remaining": int(os.environ.get("SIM_FAIL_FIRST", "0"))}


def _auth(authorization: str | None) -> None:
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="invalid token")
    if _state["fail_remaining"] > 0:
        _state["fail_remaining"] -= 1
        raise HTTPException(status_code=503, detail="no ready replicas")


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = 400
    temperature: float | None = 0.0


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    _auth(authorization)
    return {"object": "list", "data": [{"id": CHAT_MODEL, "object": "model", "owned_by": "vllm"},
                                       {"id": EMBED_MODEL, "object": "model", "owned_by": "vllm"}]}


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest, authorization: str | None = Header(default=None)):
    _auth(authorization)
    if req.model != EMBED_MODEL:
        raise HTTPException(status_code=404, detail=f"model {req.model} not served")
    texts = [req.input] if isinstance(req.input, str) else req.input
    data = [{"object": "embedding", "index": i, "embedding": embedding.embed(t)} for i, t in enumerate(texts)]
    n = sum(len(t.split()) for t in texts)
    return {"object": "list", "model": req.model, "data": data, "usage": {"prompt_tokens": n, "total_tokens": n}}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    _auth(authorization)
    if req.model != CHAT_MODEL:
        raise HTTPException(status_code=404, detail=f"model {req.model} not served")
    user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    text = extractive.answer(user)
    prompt_tokens = sum(len(m.content.split()) for m in req.messages)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(text.split()),
                  "total_tokens": prompt_tokens + len(text.split())},
    }


@app.get("/health")
def health():
    return {"status": "ok"}
