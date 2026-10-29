import pytest
from conftest import ROOT, TEST_DB
from fastapi.testclient import TestClient

from fieldassist.api import create_app
from fieldassist.settings import Settings

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture(scope="module")
def api(sim_http, indexed):
    settings = Settings(database_url=TEST_DB, chat_url="http://sim", chat_model="granite-chat",
                        embed_url="http://sim", embed_model="granite-embed", model_token="local-model-token",
                        api_token="test-token", data_dir=ROOT / "data",
                        retrieval_config=ROOT / "deploy" / "overlays" / "prod" / "retrieval.yaml")
    with TestClient(create_app(settings, http=sim_http)) as client:
        yield client


def test_ask_requires_a_bearer_token(api):
    assert api.post("/v1/ask", json={"question": "What is part 4434-3059?"}).status_code == 401


def test_ask_returns_a_cited_answer(api):
    body = api.post("/v1/ask", json={"question": "What does fault F6580 mean?"}, headers=AUTH).json()
    assert not body["refused"] and "emergency stop pressed" in body["answer"]
    assert body["citations"][0]["doc_id"] == "MAN-KR110" and body["config"] == "production"
    assert set(body["timings_ms"]) >= {"embed", "retrieve", "generate", "total"}


def test_ask_refuses_a_model_line_with_no_documentation(api):
    body = api.post("/v1/ask", json={"question": "How often should I change the oil on the TD-150?"},
                    headers=AUTH).json()
    assert body["refused"] and body["citations"] == []


def test_index_status_lists_superseded_sections(api):
    body = api.get("/v1/index/status", headers=AUTH).json()
    assert body["documents"] == {"bulletin": 8, "manual": 6}
    assert len(body["superseded_sections"]) == 8
    assert body["retrieval_config"]["mode"] == "hybrid"


def test_readiness_and_metrics(api):
    ready = api.get("/readyz").json()
    assert ready["ready"] and ready["active_chunks"] > 0
    api.post("/v1/ask", json={"question": "What is the TD-75 pressure dew point?"}, headers=AUTH)
    assert "fieldassist_ask_seconds_count" in api.get("/metrics").text
