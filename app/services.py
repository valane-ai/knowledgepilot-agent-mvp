import ast
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import run_agent, run_agent_stream
from .document_processing import process_document
from .query_rewrite import rewrite_query
from .retrieval import HybridRetriever, RetrievalChunk
from .schemas import FaithfulnessReport, Source
from .storage import SQLiteStore

DATA_DIR = Path("data")
INDEX_FILE = DATA_DIR / "index.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"
TASKS_FILE = DATA_DIR / "tasks.json"
DOCUMENTS_FILE = DATA_DIR / "documents.json"
UPLOAD_DIR = DATA_DIR / "uploads"
DEFAULT_TASK_NAME = "默认任务"
_RETRIEVER: HybridRetriever | None = None
_STORE: SQLiteStore | None = None


def load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()


@dataclass
class Chunk:
    id: str
    document: str
    content: str
    task_id: str = ""
    document_id: str = ""


def _read_json(path: Path, default):
    # ``utf-8-sig`` accepts both regular UTF-8 and files saved by Windows PowerShell with a BOM.
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None or _RETRIEVER.data_dir != DATA_DIR:
        _RETRIEVER = HybridRetriever(DATA_DIR)
    return _RETRIEVER


def _store() -> SQLiteStore:
    global _STORE
    if _STORE is None or _STORE.data_dir != DATA_DIR:
        _STORE = SQLiteStore(DATA_DIR)
        _STORE.initialize()
    return _STORE


def ensure_storage() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    _store()


def _migrate_legacy_data() -> None:
    """Assign pre-task installations to a default task once, preserving their data."""
    raw_chunks = _read_json(INDEX_FILE, [])
    if raw_chunks and all(item.get("task_id") for item in raw_chunks):
        return
    task_list = _read_json(TASKS_FILE, [])
    default_task = next((task for task in task_list if task["name"] == DEFAULT_TASK_NAME), None)
    if not default_task:
        default_task = {"id": uuid.uuid4().hex[:10], "name": DEFAULT_TASK_NAME}
        task_list.insert(0, default_task)
        _write_json(TASKS_FILE, task_list)
    document_list = _read_json(DOCUMENTS_FILE, [])
    known = {(doc["task_id"], doc["filename"]): doc for doc in document_list}
    for item in raw_chunks:
        if item.get("task_id"):
            continue
        key = (default_task["id"], item["document"])
        doc = known.get(key)
        if not doc:
            stored = next((path.name for path in UPLOAD_DIR.glob(f"*_{Path(item['document']).name}")), None)
            doc = {"id": uuid.uuid4().hex[:10], "task_id": default_task["id"], "filename": item["document"], "stored_filename": stored}
            document_list.append(doc)
            known[key] = doc
        item["task_id"] = default_task["id"]
        item["document_id"] = doc["id"]
    if raw_chunks:
        _write_json(INDEX_FILE, raw_chunks)
    _write_json(DOCUMENTS_FILE, document_list)


def load_chunks() -> list[Chunk]:
    ensure_storage()
    return [Chunk(**item) for item in _store().chunks()]


def save_chunks(chunks: list[Chunk]) -> None:
    ensure_storage()
    _write_json(INDEX_FILE, [asdict(chunk) for chunk in chunks])


def _tasks() -> list[dict]:
    ensure_storage()
    return _store().tasks()


def _documents() -> list[dict]:
    ensure_storage()
    return []


def _require_task(task_id: str) -> dict:
    task = _store().task(task_id)
    if not task:
        raise KeyError(task_id)
    return task


def tasks() -> list[dict]:
    return _store().tasks()


def create_task(name: str) -> dict:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("任务名称不能为空")
    if any(task["name"] == clean_name for task in _store().tasks()):
        raise ValueError("已存在同名任务")
    task = {"id": uuid.uuid4().hex[:10], "name": clean_name}
    _store().add_task(task["id"], task["name"])
    return {**task, "document_count": 0, "chunk_count": 0}


def _tokens(text: str) -> list[str]:
    return HybridRetriever.tokenize(text)


def split_text(text: str, size: int = 450, overlap: int = 80) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("。", start, end), text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + 150:
                end = boundary + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def extract_text(filename: str, raw: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
    return raw.decode("utf-8", errors="ignore")


def ingest(task_id: str, filename: str, raw: bytes) -> dict:
    _require_task(task_id)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    parts = process_document(filename, raw)
    if not parts:
        raise ValueError("文档中未提取到可用文本")
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(filename).name}"
    temp_path = UPLOAD_DIR / f".{safe_name}.uploading"
    stored_path = UPLOAD_DIR / safe_name
    temp_path.write_bytes(raw)
    document = {"id": uuid.uuid4().hex[:10], "task_id": task_id, "filename": Path(filename).name, "stored_filename": safe_name}
    new_chunks = [Chunk(id=uuid.uuid4().hex[:10], document=document["filename"], content=part, task_id=task_id, document_id=document["id"]) for part in parts]
    try:
        temp_path.replace(stored_path)
        _store().add_document_with_chunks(document, [asdict(chunk) for chunk in new_chunks])
    except Exception:
        temp_path.unlink(missing_ok=True)
        stored_path.unlink(missing_ok=True)
        raise
    _retriever().upsert([RetrievalChunk(**asdict(chunk)) for chunk in new_chunks])
    return {"id": document["id"], "filename": document["filename"], "chunks": len(parts)}


def queue_ingest(task_id: str, filename: str, raw: bytes) -> dict:
    """Persist an upload quickly; the caller runs process_ingest in a background task."""
    _require_task(task_id)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(filename).name}"
    temp_path = UPLOAD_DIR / f".{safe_name}.uploading"
    stored_path = UPLOAD_DIR / safe_name
    document = {"id": uuid.uuid4().hex[:10], "task_id": task_id, "filename": Path(filename).name, "stored_filename": safe_name}
    try:
        temp_path.write_bytes(raw)
        temp_path.replace(stored_path)
        _store().create_processing_document(document)
    except Exception:
        temp_path.unlink(missing_ok=True)
        stored_path.unlink(missing_ok=True)
        raise
    return {"id": document["id"], "filename": document["filename"], "chunks": 0, "status": "processing"}


def process_ingest(task_id: str, document_id: str) -> None:
    """Background parser/indexer. Failure is represented as a durable document status."""
    document = _store().document(task_id, document_id)
    if not document:
        return
    try:
        raw = (UPLOAD_DIR / document["stored_filename"]).read_bytes()
        _store().set_document_status(document_id, "processing", progress=35)
        parts = process_document(document["filename"], raw)
        if not parts:
            raise ValueError("文档中未提取到可用文本")
        chunks = [Chunk(id=uuid.uuid4().hex[:10], document=document["filename"], content=part, task_id=task_id, document_id=document_id) for part in parts]
        _store().set_document_status(document_id, "processing", progress=80)
        _store().complete_document(document_id, [asdict(chunk) for chunk in chunks])
        _retriever().upsert([RetrievalChunk(**asdict(chunk)) for chunk in chunks])
    except Exception as exc:
        _store().set_document_status(document_id, "failed", error_detail=str(exc)[:500])


def reindex_document(task_id: str, document_id: str) -> dict:
    _require_task(task_id)
    document = _store().start_reindex(task_id, document_id)
    if not document:
        raise KeyError(document_id)
    _retriever().rebuild_task(task_id, [RetrievalChunk(**item) for item in _store().chunks(task_id)])
    return {"id": document["id"], "filename": document["filename"], "chunks": 0, "status": "processing"}


def delete_task(task_id: str) -> None:
    files = _store().delete_task(task_id)
    if files is None:
        raise KeyError(task_id)
    _retriever().rebuild_task(task_id, [])
    for filename in files:
        (UPLOAD_DIR / filename).unlink(missing_ok=True)


def documents(task_id: str) -> list[dict]:
    _require_task(task_id)
    return _store().documents(task_id)


def delete_document(task_id: str, document_id: str) -> None:
    _require_task(task_id)
    document = _store().delete_document_with_chunks(task_id, document_id)
    if not document:
        raise KeyError(document_id)
    remaining_chunks = [Chunk(**item) for item in _store().chunks(task_id)]
    _retriever().rebuild_task(task_id, [RetrievalChunk(**asdict(chunk)) for chunk in remaining_chunks])
    if document.get("stored_filename"):
        stored_path = UPLOAD_DIR / document["stored_filename"]
        if stored_path.exists():
            stored_path.unlink()


def _metadata_filter_for_task(task_id: str, query: str, requested: dict | None = None) -> dict | None:
    """Validate requested filters and infer a document filter when its name is in the question."""
    available = _store().documents(task_id)
    by_id = {item["id"]: item["filename"] for item in available}
    names = set(by_id.values())
    requested = requested or {}
    document_ids = [item for item in requested.get("document_ids", []) if item in by_id]
    documents = [item for item in requested.get("documents", []) if item in names]
    lowered_query = query.lower()
    inferred = [item["filename"] for item in available if len(Path(item["filename"]).stem) >= 3 and Path(item["filename"]).stem.lower() in lowered_query]
    documents = list(dict.fromkeys(documents + inferred))
    result = {}
    if document_ids:
        result["document_ids"] = document_ids
    if documents:
        result["documents"] = documents
    return result or None


def search_with_trace(query: str, task_id: str, top_k: int | None = None, metadata_filter: dict | None = None) -> tuple[list[Source], list[str]]:
    retriever = _retriever()
    # Old JSON-only installations are backfilled once. Subsequent queries use persisted indexes.
    if not retriever.has_task(task_id):
        task_chunks = [RetrievalChunk(**asdict(chunk)) for chunk in load_chunks() if chunk.task_id == task_id]
        retriever.rebuild_task(task_id, task_chunks)
    rewritten_query, rewrite_trace = rewrite_query(query)
    validated_filter = _metadata_filter_for_task(task_id, query, metadata_filter)
    results, trace = retriever.search(rewritten_query, task_id, top_k, validated_filter)
    return [Source(document=item["document"], chunk_id=item["chunk_id"], content=item["content"], score=item["score"]) for item in results], [rewrite_trace] + trace if rewrite_trace else trace


def search(query: str, task_id: str, top_k: int | None = None, metadata_filter: dict | None = None) -> list[Source]:
    return search_with_trace(query, task_id, top_k, metadata_filter)[0]


def _safe_calculate(expression: str) -> str:
    if len(expression) > 200:
        raise ValueError("计算表达式过长")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("计算表达式过于复杂")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = evaluate(node.operand)
            value = operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            elif isinstance(node.op, ast.Mult): value = left * right
            elif isinstance(node.op, ast.Div): value = left / right
            elif isinstance(node.op, ast.Mod): value = left % right
            else:
                if abs(right) > 10:
                    raise ValueError("幂运算指数不能超过 10")
                value = left ** right
        else:
            raise ValueError("仅支持数字、括号和基础四则运算")
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("计算结果超出允许范围")
        return value

    return str(evaluate(tree.body))


def _deprecated_llm_answer(question: str, sources: list[Source]) -> tuple[str | None, str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "未检测到 OPENAI_API_KEY"
    context = "\n\n".join(f"[{source.document}] {source.content}" for source in sources)
    system = "你是企业知识库助手。优先根据给定资料回答，并明确资料不足之处。若没有提供资料，可基于通用知识回答，但不要把通用知识伪装成资料结论。使用简洁中文。"
    user = f"问题：{question}\n\n" + (f"资料：\n{context}" if context else "本次检索没有命中资料，请直接基于通用知识回答，并说明该回答没有引用企业资料。")
    payload = {"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0.2}
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    request = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"].strip(), None
    except urllib.error.HTTPError as exc:
        return None, f"模型服务返回 HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}"
    except urllib.error.URLError as exc:
        return None, f"无法连接模型服务: {exc.reason}"
    except TimeoutError:
        return None, "连接模型服务超时（20 秒）"
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return None, f"模型响应格式不符合 OpenAI 兼容协议: {exc}"


def _load_sessions() -> dict:
    return {}


def get_session(session_id: str, task_id: str | None = None) -> list[dict]:
    ensure_storage()
    return _store().messages(session_id, task_id)


def _append_message(session_id: str, role: str, content: str, task_id: str | None = None) -> None:
    ensure_storage()
    _store().append_message(session_id, role, content, task_id)


def _legacy_answer(message: str, task_id: str, session_id: str | None = None) -> dict:
    _require_task(task_id)
    session_id = session_id or uuid.uuid4().hex
    trace = ["收到用户问题", "已限定在当前任务的资料范围内"]
    _append_message(session_id, "user", message, task_id)
    calculation = re.fullmatch(r"\s*(?:计算|calc)?\s*([0-9+\-*/%().\s]+)\s*", message, flags=re.I)
    if calculation:
        try:
            reply = f"计算结果：{_safe_calculate(calculation.group(1))}"
            route, sources = "calculator", []
            trace.extend(["选择 calculator 工具", "工具执行成功"])
        except Exception as exc:
            reply, route, sources = f"无法计算：{exc}", "calculator", []
    else:
        sources, retrieval_trace = search_with_trace(message, task_id)
        trace.extend(retrieval_trace)
        trace.append(f"当前任务混合检索完成，命中 {len(sources)} 个片段")
        generated, llm_error = _deprecated_llm_answer(message, sources)
        if generated:
            reply = generated
            route = "knowledge_base" if sources else "general_llm"
            trace.append("调用 LLM 生成答案")
        elif sources:
            evidence = "\n\n".join(f"【{source.document}】{source.content}" for source in sources)
            reply = f"模型调用失败，以下是当前任务中相关的资料原文：\n\n{evidence}"
            route = "knowledge_base"
            trace.append(f"LLM 调用失败，使用本地资料降级回答：{llm_error}")
        else:
            reply = f"当前任务没有命中相关资料，且无法调用大模型：{llm_error}。请配置模型服务后重试。"
            route = "fallback"
            trace.append(f"未命中资料，LLM 调用失败：{llm_error}")
    _append_message(session_id, "assistant", reply, task_id)
    trace.append("生成回答完成")
    return {"session_id": session_id, "answer": reply, "sources": sources, "route": route, "trace": trace}


def _local_rag_fallback(message: str, task_id: str, trace: list[str], llm_error: str) -> tuple[str, list[Source], str]:
    """Keep the application useful when an API key or tool-calling provider is unavailable."""
    sources, retrieval_trace = search_with_trace(message, task_id)
    trace.extend(retrieval_trace)
    if sources:
        evidence = "\n\n".join(f"[{source.document}#{source.chunk_id}] {source.content}" for source in sources)
        return f"模型 Agent 不可用，以下是当前任务中的相关资料原文：\n\n{evidence}", sources, "knowledge_base"
    return f"当前任务没有命中相关资料，且无法调用模型 Agent：{llm_error}", [], "fallback"


GENERAL_KNOWLEDGE_NOTICE = "提示：以下回答基于通用知识，没有内部资料参考。"
_CITATION_PATTERN = re.compile(r"\[([^\[\]#]+#[A-Za-z0-9_-]+)\]")


def validate_faithfulness(answer: str, sources: list[Source]) -> FaithfulnessReport:
    """Validate citation coverage against the sources returned in this request.

    This is a deterministic grounding check, not an NLI claim-entailment model: it catches
    invented/missing citations and exposes longer uncited statements for human review.
    """
    if not sources:
        return FaithfulnessReport(
            status="not_applicable",
            message="本回答未引用内部资料；如为通用回答，已在答案中明确标注。",
        )
    valid = {f"{source.document}#{source.chunk_id}" for source in sources}
    citations = _CITATION_PATTERN.findall(answer)
    invalid = sorted(set(citation for citation in citations if citation not in valid))
    statements = [part.strip() for part in re.split(r"[。！？!?\n]+", answer) if part.strip()]
    uncited = []
    for statement in statements:
        text_without_citations = _CITATION_PATTERN.sub("", statement).strip()
        if len(text_without_citations) >= 16 and not _CITATION_PATTERN.search(statement):
            uncited.append(statement[:120])
    if invalid:
        status = "warning"
        message = "发现答案引用了本次检索结果之外的片段，请核对后再采信。"
    elif not citations:
        status = "warning"
        message = "检索到了内部资料，但答案没有提供内联引用。"
    elif uncited:
        status = "warning"
        message = "部分较长陈述没有内联引用，建议人工核对。"
    else:
        status = "supported"
        message = "所有较长陈述均带有本次检索来源的有效内联引用。"
    return FaithfulnessReport(
        status=status,
        message=message,
        citations=sorted(set(citations)),
        invalid_citations=invalid,
        uncited_statements=uncited,
    )


def answer(message: str, task_id: str, session_id: str | None = None) -> dict:
    """Answer through a bounded, stateful LLM tool-calling loop instead of regex routing."""
    _require_task(task_id)
    session_id = session_id or uuid.uuid4().hex
    _append_message(session_id, "user", message, task_id)
    history = get_session(session_id, task_id)
    reply, sources, trace, route, llm_error = run_agent(
        session_id=session_id,
        task_id=task_id,
        history=history,
        search_fn=search_with_trace,
        calculate_fn=_safe_calculate,
    )
    if reply is None:
        fallback_trace = ["Agent 调用失败，使用本地降级策略"] + trace
        reply, sources, route = _local_rag_fallback(message, task_id, fallback_trace, llm_error or "未知错误")
        trace = fallback_trace
    if not sources and route in {"general_llm", "fallback"} and not reply.startswith(GENERAL_KNOWLEDGE_NOTICE):
        reply = f"{GENERAL_KNOWLEDGE_NOTICE}\n\n{reply}"
    faithfulness = validate_faithfulness(reply, sources)
    trace.append(f"引用一致性校验：{faithfulness.status}")
    _append_message(session_id, "assistant", reply, task_id)
    trace.append("生成回答完成")
    return {"session_id": session_id, "answer": reply, "sources": sources, "faithfulness": faithfulness, "route": route, "trace": trace}


def stream_answer(message: str, task_id: str, session_id: str | None = None):
    """Yield SSE-ready event dictionaries and persist the completed assistant reply."""
    _require_task(task_id)
    session_id = session_id or uuid.uuid4().hex
    _append_message(session_id, "user", message, task_id)
    history = get_session(session_id, task_id)
    reply_parts: list[str] = []
    started = False
    final_sources: list[Source] = []
    final_trace: list[str] = []
    route = "agent"

    for event in run_agent_stream(session_id, task_id, history, search_with_trace, _safe_calculate):
        if event["type"] == "start":
            started = True
            if not event["has_sources"]:
                notice = f"{GENERAL_KNOWLEDGE_NOTICE}\n\n"
                reply_parts.append(notice)
                yield {"event": "token", "data": {"content": notice}}
        elif event["type"] == "token":
            reply_parts.append(event["content"])
            yield {"event": "token", "data": {"content": event["content"]}}
        elif event["type"] == "done":
            final_sources = event["sources"]
            final_trace = event["trace"]
            route = event["route"]
        elif event["type"] == "error":
            final_sources = event.get("sources", [])
            final_trace = ["Agent 流式调用失败，使用本地降级策略"] + event.get("trace", [])
            reply, final_sources, route = _local_rag_fallback(message, task_id, final_trace, event["error"])
            if not final_sources and not reply.startswith(GENERAL_KNOWLEDGE_NOTICE):
                reply = f"{GENERAL_KNOWLEDGE_NOTICE}\n\n{reply}"
            if not started:
                yield {"event": "token", "data": {"content": reply}}
                reply_parts.append(reply)

    reply = "".join(reply_parts)
    if not reply:
        reply = "模型未返回可展示的回答。"
        yield {"event": "token", "data": {"content": reply}}
    faithfulness = validate_faithfulness(reply, final_sources)
    final_trace.append(f"引用一致性校验：{faithfulness.status}")
    _append_message(session_id, "assistant", reply, task_id)
    yield {
        "event": "done",
        "data": {
            "session_id": session_id,
            "sources": [source.model_dump() for source in final_sources],
            "faithfulness": faithfulness.model_dump(),
            "route": route,
            "trace": final_trace,
        },
    }
