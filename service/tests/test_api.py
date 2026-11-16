"""Uji API tanpa database dan tanpa LLM: engine RAG diganti mock."""
from fastapi.testclient import TestClient

from service.app.engine import Result, Source
from service.app.main import create_app
from service.app.settings import Settings


class FakeEngine:
    def __init__(self, ready=True):
        self._ready = ready
        self.calls = []

    async def ready(self):
        return self._ready

    async def query(self, question, role):
        self.calls.append((question, role))
        if "gagal" in question:
            raise RuntimeError("LLM mati")
        if "rendang" in question:
            return Result("di luar cakupan", blocked=True, flags=["di_luar_cakupan"], top_score=0.2)
        return Result("Jawaban [1].", model="uji/palsu", top_score=0.7, latency_ms=5,
                      sources=[Source("arxiv:1", "Judul", 0.7, "isi chunk yang tidak boleh ikut keluar")])


def make_client(engine=None):
    settings = Settings(openrouter_api_key="uji", service_api_keys="k-pub:public,k-int:internal")
    return TestClient(create_app(engine=engine or FakeEngine(), settings=settings))


def test_health_and_ready():
    with make_client() as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 200


def test_not_ready_when_library_is_empty():
    with make_client(FakeEngine(ready=False)) as client:
        assert client.get("/readyz").status_code == 503


def test_query_requires_known_api_key():
    with make_client() as client:
        assert client.post("/v1/query", json={"question": "apa itu RAG?"}).status_code == 401
        response = client.post("/v1/query", json={"question": "apa itu RAG?"}, headers={"X-API-Key": "salah"})
        assert response.status_code == 401


def test_role_comes_from_api_key_not_from_body():
    engine = FakeEngine()
    with make_client(engine) as client:
        client.post("/v1/query", json={"question": "anggaran lab?", "role": "internal"},
                    headers={"X-API-Key": "k-pub"})
        client.post("/v1/query", json={"question": "anggaran lab?"}, headers={"X-API-Key": "k-int"})
    assert [role for _, role in engine.calls] == ["public", "internal"]


def test_response_shape_and_request_id():
    with make_client() as client:
        response = client.post("/v1/query", json={"question": "apa itu RAG?"},
                               headers={"X-API-Key": "k-pub", "X-Request-ID": "abc123"})
    body = response.json()
    assert response.status_code == 200
    assert body["request_id"] == "abc123"
    assert body["sources"] == [{"source_id": "arxiv:1", "title": "Judul", "score": 0.7}]
    assert "isi chunk" not in response.text  # teks chunk tidak dikirim ke client


def test_blocked_request_is_reported():
    with make_client() as client:
        body = client.post("/v1/query", json={"question": "resep rendang"}, headers={"X-API-Key": "k-pub"}).json()
    assert body["blocked"] is True and body["flags"] == ["di_luar_cakupan"]


def test_engine_failure_becomes_502_without_leaking_details():
    with make_client() as client:
        response = client.post("/v1/query", json={"question": "gagal"}, headers={"X-API-Key": "k-pub"})
    assert response.status_code == 502
    assert "LLM mati" not in response.text


def test_question_length_is_validated():
    with make_client() as client:
        response = client.post("/v1/query", json={"question": "a" * 2001}, headers={"X-API-Key": "k-pub"})
    assert response.status_code == 422


def test_metrics_are_exposed():
    with make_client() as client:
        client.post("/v1/query", json={"question": "apa itu RAG?"}, headers={"X-API-Key": "k-pub"})
        assert "rag_requests_total" in client.get("/metrics").text


def test_settings_require_api_keys(monkeypatch):
    import pytest
    from pydantic import ValidationError

    monkeypatch.delenv("SERVICE_API_KEYS", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openrouter_api_key="uji")


def test_demo_keys_are_detected():
    demo = Settings(_env_file=None, openrouter_api_key="uji", service_api_keys="kunci-publik-demo:public")
    real = Settings(_env_file=None, openrouter_api_key="uji", service_api_keys="k-pub:public")
    assert demo.uses_demo_keys and not real.uses_demo_keys
