"""Client for the two KServe InferenceServices, both vLLM speaking the OpenAI-compatible API.

During a rollout, or while a replica is still loading weights, the KServe route answers 503, so
the client retries 502/503/504 and connection errors with jittered exponential backoff inside a
fixed deadline. A 4xx is a bug on our side and is raised at once.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import httpx

RETRY_STATUS = {502, 503, 504}


class ModelError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class ModelClient:
    def __init__(self, base_url: str, model: str, token: str, http: httpx.Client | None = None,
                 attempts: int = 4, base_delay: float = 0.5, deadline: float = 45.0,
                 sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http = http or httpx.Client(timeout=httpx.Timeout(30.0, connect=2.0))
        self.headers = {"Authorization": f"Bearer {token}"}
        self.attempts = attempts
        self.base_delay = base_delay
        self.deadline = deadline
        self.sleep = sleep

    def _post(self, path: str, payload: dict) -> dict:
        start = time.monotonic()
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                resp = self.http.post(f"{self.base_url}{path}", json=payload, headers=self.headers)
                if resp.status_code in RETRY_STATUS:
                    last = ModelError(f"{path} returned {resp.status_code}")
                else:
                    if resp.status_code >= 400:
                        raise ModelError(f"{path} returned {resp.status_code}: {resp.text[:200]}")
                    return resp.json()
            except httpx.TransportError as exc:  # connect refused or timed out, reset, read timeout
                last = exc
            delay = self.base_delay * (2 ** attempt) * random.uniform(0.8, 1.2)
            if attempt == self.attempts - 1 or time.monotonic() - start + delay > self.deadline:
                break
            self.sleep(delay)
        raise ModelError(f"{path} failed after retries: {last}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            body = self._post("/v1/embeddings", {"model": self.model, "input": texts[i:i + 64]})
            rows = sorted(body["data"], key=lambda d: d["index"])
            out.extend(r["embedding"] for r in rows)
        return out

    def chat(self, messages: list[dict], max_tokens: int = 400) -> ChatResult:
        t0 = time.monotonic()
        body = self._post("/v1/chat/completions", {
            "model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
        })
        usage = body.get("usage") or {}
        return ChatResult(
            text=body["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            seconds=time.monotonic() - t0,
        )

    def ready(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/v1/models", headers=self.headers)
            return resp.status_code == 200 and any(m["id"] == self.model for m in resp.json()["data"])
        except (httpx.HTTPError, KeyError, ValueError):
            return False
