from app import services
from app import agent
from app.agent import run_agent
from app.document_processing import process_document
from app.retrieval import HybridRetriever
from app.schemas import Source
import pytest


@pytest.fixture(autouse=True)
def disable_local_models(monkeypatch):
    """Unit tests must not download embedding/reranker models."""
    monkeypatch.setattr(HybridRetriever, "_init_vector_backend", lambda self: False)
    monkeypatch.setattr(HybridRetriever, "_rerank", lambda self, query, candidates, chunks: candidates)


def test_split_text_keeps_content():
    assert services.split_text("这是一个用于测试的知识库段落。") == ["这是一个用于测试的知识库段落。"]


def test_calculator_accepts_math():
    assert services._safe_calculate("(12+8)*3") == "60"


def test_calculator_rejects_code_and_excessive_exponent():
    with pytest.raises(ValueError):
        services._safe_calculate("__import__('os').system('whoami')")
    with pytest.raises(ValueError):
        services._safe_calculate("2 ** 11")


def test_tasks_isolate_documents_and_support_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "DATA_DIR", tmp_path)
    monkeypatch.setattr(services, "INDEX_FILE", tmp_path / "index.json")
    monkeypatch.setattr(services, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(services, "TASKS_FILE", tmp_path / "tasks.json")
    monkeypatch.setattr(services, "DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")

    first_task = services.create_task("First task")
    second_task = services.create_task("Second task")
    document = services.ingest(first_task["id"], "first.txt", b"alpha project handbook")
    services.ingest(second_task["id"], "second.txt", b"beta product handbook")

    assert services.search("alpha", first_task["id"])
    assert not services.search("alpha", second_task["id"])
    services.delete_document(first_task["id"], document["id"])
    assert services.documents(first_task["id"]) == []


def test_search_uses_persisted_bm25_index_after_ingestion(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "DATA_DIR", tmp_path)
    monkeypatch.setattr(services, "INDEX_FILE", tmp_path / "index.json")
    monkeypatch.setattr(services, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(services, "TASKS_FILE", tmp_path / "tasks.json")
    monkeypatch.setattr(services, "DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    task = services.create_task("Persistent retrieval")
    services.ingest(task["id"], "manual.txt", b"semantic retrieval architecture")

    assert (tmp_path / "bm25_index.json").exists()
    monkeypatch.setattr(services, "load_chunks", lambda: (_ for _ in ()).throw(AssertionError("query should not scan index.json")))
    assert services.search("retrieval", task["id"])


def test_query_rewrite_and_filename_metadata_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "DATA_DIR", tmp_path)
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(services, "rewrite_query", lambda question: ("annual leave policy", "Query Rewrite: annual leave policy"))
    task = services.create_task("Policies")
    services.ingest(task["id"], "leave-policy.md", b"annual leave policy requires approval")
    services.ingest(task["id"], "expense-policy.md", b"expense reimbursement policy")

    sources, trace = services.search_with_trace("请查询 leave-policy 的规定", task["id"])

    assert sources and {source.document for source in sources} == {"leave-policy.md"}
    assert "Query Rewrite" in trace[0]
    assert any("元数据过滤" in item for item in trace)


def test_dynamic_top_k_limits_ambiguous_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DYNAMIC_MAX_K", "3")
    retriever = HybridRetriever(tmp_path)
    chunks = [
        services.RetrievalChunk(id=f"chunk-{index}", task_id="task", document_id=f"doc-{index}", document=f"doc-{index}.md", content="annual leave policy")
        for index in range(5)
    ]
    retriever.upsert(chunks)

    results, trace = retriever.search("annual leave", "task")

    assert 1 <= len(results) <= 3
    assert any("动态 top-k" in item for item in trace)


def test_background_ingest_transitions_document_to_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "DATA_DIR", tmp_path)
    monkeypatch.setattr(services, "INDEX_FILE", tmp_path / "index.json")
    monkeypatch.setattr(services, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(services, "TASKS_FILE", tmp_path / "tasks.json")
    monkeypatch.setattr(services, "DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    task = services.create_task("Background ingestion")
    document = services.queue_ingest(task["id"], "queued.txt", b"background indexing works")

    assert document["status"] == "processing"
    services.process_ingest(task["id"], document["id"])
    stored = services.documents(task["id"])[0]
    assert stored["status"] == "ready"
    assert stored["chunks"] == 1


def test_semantic_chunking_keeps_markdown_heading_context():
    parts = process_document(
        "policy.md",
        "# Leave policy\n\nAnnual leave is paid.\n\n## Eligibility\n\nEmployees qualify after probation.".encode(),
    )
    assert any(part.startswith("Leave policy") for part in parts)
    assert any(part.startswith("Eligibility") for part in parts)


def test_delete_task_cascades_documents_and_task_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "DATA_DIR", tmp_path)
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    task = services.create_task("Disposable task")
    services.ingest(task["id"], "manual.txt", b"temporary content")
    services._append_message("session-1", "user", "temporary question", task["id"])

    services.delete_task(task["id"])

    assert not services.tasks()
    assert not services._store().chunks(task["id"])
    assert not services.get_session("session-1")


def test_no_hit_still_uses_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "DATA_DIR", tmp_path)
    monkeypatch.setattr(services, "INDEX_FILE", tmp_path / "index.json")
    monkeypatch.setattr(services, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(services, "TASKS_FILE", tmp_path / "tasks.json")
    monkeypatch.setattr(services, "DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    task = services.create_task("LLM task")
    monkeypatch.setattr(
        services,
        "run_agent",
        lambda **kwargs: ("general answer", [], ["LLM generated directly"], "general_llm", None),
    )

    result = services.answer("question without a matching document", task["id"])
    assert result["route"] == "general_llm"
    assert result["answer"].endswith("general answer")
    assert result["answer"].startswith(services.GENERAL_KNOWLEDGE_NOTICE)
    assert result["faithfulness"].status == "not_applicable"


def test_faithfulness_checks_inline_citation_validity():
    source = Source(document="policy.md", chunk_id="chunk_1", content="Annual leave policy", score=0.9)
    report = services.validate_faithfulness("Employees receive annual leave [policy.md#chunk_1]", [source])
    assert report.status == "supported"
    assert report.citations == ["policy.md#chunk_1"]

    invalid = services.validate_faithfulness("Employees receive annual leave [policy.md#missing]", [source])
    assert invalid.status == "warning"
    assert invalid.invalid_citations == ["policy.md#missing"]


def test_agent_passes_history_and_completes_a_search_tool_call():
    calls = []
    responses = iter(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_knowledge_base", "arguments": '{"query":"leave policy"}'},
                    }
                ],
            },
            {"role": "assistant", "content": "According to the policy, annual leave is available.", "tool_calls": []},
        ]
    )

    def fake_completion(messages):
        calls.append(messages)
        return next(responses), None

    source = Source(document="policy.md", chunk_id="chunk_1", content="Annual leave policy", score=0.9)
    result = run_agent(
        session_id="session_1",
        task_id="task_1",
        history=[
            {"role": "user", "content": "What is annual leave?"},
            {"role": "assistant", "content": "I can help with policies."},
            {"role": "user", "content": "Please check the current policy."},
        ],
        search_fn=lambda query, task_id: ([source], ["retrieval completed"]),
        calculate_fn=services._safe_calculate,
        completion_fn=fake_completion,
    )

    answer, sources, trace, route, error = result
    assert error is None
    assert answer.startswith("According")
    assert route == "agent_knowledge"
    assert sources == [source]
    assert any(message["content"] == "What is annual leave?" for message in calls[0])
    assert any(message["role"] == "tool" and message["tool_call_id"] == "call_1" for message in calls[1])
    assert "执行工具：search_knowledge_base" in trace


def test_streaming_agent_yields_tokens_after_tool_call(monkeypatch):
    responses = iter(
        [
            [{"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "search_knowledge_base", "arguments": '{"query":"leave"}'}}]}],
            [{"content": "Annual leave is available."}],
        ]
    )
    monkeypatch.setattr(agent, "stream_chat_completion", lambda payload: iter(next(responses)))
    source = Source(document="policy.md", chunk_id="chunk_1", content="Annual leave policy", score=0.9)

    events = list(agent.run_agent_stream("s1", "t1", [{"role": "user", "content": "leave?"}], lambda query, task: ([source], ["retrieval"]), services._safe_calculate))

    assert any(event["type"] == "token" and "Annual leave" in event["content"] for event in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["sources"] == [source]
