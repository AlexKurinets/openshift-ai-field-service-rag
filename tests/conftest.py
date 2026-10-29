"""Shared fixtures: a real Postgres with pgvector, the model stand-in served in-process, and the full
synthetic corpus ingested once per session."""

import os
from pathlib import Path

import psycopg
import pytest
from starlette.testclient import TestClient

from fieldassist import corpus, evaluation
from fieldassist.db import apply_schema, make_pool
from fieldassist.ingest import ingest_dir
from fieldassist.model_client import ModelClient
from fieldassist.settings import RetrievalConfig
from model_sim.app import app as sim_app

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEST_DB = os.environ.get("TEST_DATABASE_URL",
                         "postgresql://fieldassist:fieldassist@localhost:5496/fieldassist_test")
FAKE_IMAGE = "oci://quay.example.com/fieldassist/granite-3.3-8b-instruct@sha256:" + "a" * 64


def _ensure_database(url: str) -> None:
    info = psycopg.conninfo.conninfo_to_dict(url)
    name = info["dbname"]
    admin = psycopg.conninfo.make_conninfo(url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            conn.execute(f'CREATE DATABASE "{name}"')


@pytest.fixture(scope="session")
def pool():
    _ensure_database(TEST_DB)
    p = make_pool(TEST_DB)
    with p.connection() as conn:
        conn.execute("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;")
    apply_schema(p)
    yield p
    p.close()


@pytest.fixture(scope="session")
def sim_http():
    with TestClient(sim_app) as client:
        yield client


@pytest.fixture(scope="session")
def chat(sim_http):
    return ModelClient("http://sim", "granite-chat", "local-model-token", sim_http)


@pytest.fixture(scope="session")
def embedder(sim_http):
    return ModelClient("http://sim", "granite-embed", "local-model-token", sim_http)


@pytest.fixture(scope="session")
def indexed(pool, embedder):
    return ingest_dir(pool, embedder, DATA / "corpus")


@pytest.fixture(scope="session")
def golden():
    return corpus.load_golden_set(DATA / "golden_set.jsonl")


def cfg(name: str) -> RetrievalConfig:
    return RetrievalConfig.load(ROOT / "config" / "retrieval" / f"{name}.yaml")


@pytest.fixture(scope="session")
def reports(pool, chat, embedder, indexed, golden):
    """One golden-set run per configuration, shared by the gate and promotion tests."""
    return {name: evaluation.run(pool, chat, embedder, cfg(name), golden, DATA / "golden_set.jsonl", FAKE_IMAGE)
            for name in ("production", "vector-only", "no-revision-filter")}
