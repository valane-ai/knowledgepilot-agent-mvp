from fastapi.testclient import TestClient
import pytest

from app import main, services
from app.retrieval import HybridRetriever
from app.security import InMemoryRateLimiter


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("KNOWLEDGEPILOT_API_KEYS", raising=False)
    monkeypatch.setattr(services, "DATA_DIR", tmp_path)
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(HybridRetriever, "_init_vector_backend", lambda self: False)
    monkeypatch.setattr(HybridRetriever, "_rerank", lambda self, query, candidates, chunks: candidates)
    services._STORE = None
    services._RETRIEVER = None
    return TestClient(main.app)


def test_task_upload_document_and_delete_workflow(client):
    task = client.post("/api/tasks", json={"name": "API test"}).json()
    upload = client.post("/api/documents", data={"task_id": task["id"]}, files={"file": ("policy.md", b"# Policy\nAnnual leave is paid.", "text/markdown")})
    assert upload.status_code == 200
    documents = client.get(f"/api/tasks/{task['id']}/documents").json()
    assert documents[0]["status"] == "ready"
    assert documents[0]["chunks"] == 1
    assert client.delete(f"/api/tasks/{task['id']}").status_code == 204
    assert client.get("/api/tasks").json() == []


def test_api_key_is_optional_but_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("KNOWLEDGEPILOT_API_KEYS", "test-secret")
    assert client.get("/api/tasks").status_code == 401
    assert client.get("/api/tasks", headers={"X-API-Key": "test-secret"}).status_code == 200


def test_chat_rejects_control_characters(client):
    task = client.post("/api/tasks", json={"name": "Validation"}).json()
    response = client.post("/api/chat", json={"task_id": task["id"], "message": "hello\u0000world"})
    assert response.status_code == 422


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(main, "rate_limiter", InMemoryRateLimiter())
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1")
    assert client.get("/api/tasks").status_code == 200
    assert client.get("/api/tasks").status_code == 429


def test_stream_endpoint_emits_sse_events(client, monkeypatch):
    task = client.post("/api/tasks", json={"name": "Streaming"}).json()
    monkeypatch.setattr(main, "stream_answer", lambda message, task_id, session_id: iter([
        {"event": "token", "data": {"content": "hello"}},
        {"event": "done", "data": {"session_id": "s1", "sources": [], "faithfulness": {"status": "not_applicable", "message": "", "citations": [], "invalid_citations": [], "uncited_statements": []}, "route": "general_llm", "trace": []}},
    ]))
    response = client.post("/api/chat/stream", json={"task_id": task["id"], "message": "hello"})
    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: done" in response.text
