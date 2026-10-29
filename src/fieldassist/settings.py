"""Runtime settings, read from FA_* environment variables.

In the cluster these come from the gateway Deployment (deploy/base/gateway.yaml): the model URLs are
the KServe InferenceService endpoints and the token is the service account token Authorino checks.
Locally they point at the model stand-in in docker-compose.yml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    database_url: str
    chat_url: str
    chat_model: str
    embed_url: str
    embed_model: str
    model_token: str
    api_token: str
    retrieval_config: Path
    data_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        return cls(
            database_url=e("FA_DATABASE_URL", "postgresql://fieldassist:fieldassist@localhost:5496/fieldassist"),
            chat_url=e("FA_CHAT_URL", "http://localhost:8171"),
            chat_model=e("FA_CHAT_MODEL", "granite-chat"),
            embed_url=e("FA_EMBED_URL", "http://localhost:8171"),
            embed_model=e("FA_EMBED_MODEL", "granite-embed"),
            model_token=e("FA_MODEL_TOKEN", "local-model-token"),
            api_token=e("FA_API_TOKEN", "local-dev-token"),
            retrieval_config=Path(e("FA_RETRIEVAL_CONFIG", "config/retrieval/production.yaml")),
            data_dir=Path(e("FA_DATA_DIR", "data")),
        )


@dataclass(frozen=True)
class RetrievalConfig:
    """What the golden-set gate evaluates and promotes, alongside the model image."""

    name: str
    mode: str = "hybrid"  # hybrid | vector | keyword
    active_only: bool = True
    candidates_per_retriever: int = 20
    top_k: int = 5
    rrf_k: int = 60
    min_similarity: float = 0.2
    prompt_version: str = "v3"
    max_tokens: int = 400

    @classmethod
    def load(cls, path: Path | str) -> RetrievalConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cfg = cls(**raw)
        if cfg.mode not in {"hybrid", "vector", "keyword"}:
            raise ValueError(f"unknown retrieval mode {cfg.mode!r}")
        return cfg

    def as_dict(self) -> dict:
        return dict(self.__dict__)
