import httpx
import pytest

from fieldassist.model_client import ModelClient, ModelError

OK_CHAT = {"choices": [{"message": {"content": "ok [P1]"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def client(handler, attempts=4):
    sleeps = []
    c = ModelClient("http://m", "granite-chat", "t", httpx.Client(transport=httpx.MockTransport(handler)),
                    attempts=attempts, sleep=sleeps.append)
    return c, sleeps


def test_503_during_rollout_is_retried_with_backoff():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json=OK_CHAT)

    c, sleeps = client(handler)
    assert c.chat([{"role": "user", "content": "q"}]).text == "ok [P1]"
    assert len(calls) == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0]
    assert calls[0].headers["authorization"] == "Bearer t"


def test_gives_up_after_the_last_attempt():
    c, sleeps = client(lambda req: httpx.Response(503), attempts=3)
    with pytest.raises(ModelError, match="failed after retries"):
        c.chat([{"role": "user", "content": "q"}])
    assert len(sleeps) == 2


def test_client_errors_are_not_retried():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(400, text="bad request")

    c, _ = client(handler)
    with pytest.raises(ModelError, match="400"):
        c.chat([{"role": "user", "content": "q"}])
    assert len(calls) == 1


def test_connection_errors_are_retried():
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=OK_CHAT)

    c, _ = client(handler)
    assert c.chat([{"role": "user", "content": "q"}]).completion_tokens == 2


def test_embeddings_are_batched_and_reordered_by_index():
    batches = []

    def handler(req):
        inputs = __import__("json").loads(req.content)["input"]
        batches.append(len(inputs))
        data = [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(inputs)]
        return httpx.Response(200, json={"data": list(reversed(data))})

    c, _ = client(handler)
    texts = ["x" * n for n in range(1, 101)]
    out = c.embed(texts)
    assert batches == [64, 36]
    assert [v[0] for v in out] == [float(n) for n in range(1, 101)]
