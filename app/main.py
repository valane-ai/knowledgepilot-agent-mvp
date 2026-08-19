import os
import json
import zipfile
import io
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .schemas import ChatRequest, ChatResponse, DocumentInfo, SessionMessage, TaskCreate, TaskInfo
from .security import client_key, configured_limit, rate_limiter, require_api_key
from .services import answer, create_task, delete_document, delete_task, documents, get_session, process_ingest, queue_ingest, reindex_document, stream_answer, tasks

app = FastAPI(title="KnowledgePilot Agent", version="0.3.0")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def protect_api(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        try:
            require_api_key(request)
            if request.url.path.startswith("/api/chat"):
                rate_limiter.check(f"chat:{client_key(request)}", configured_limit("CHAT_RATE_LIMIT_PER_MINUTE", 20))
            elif request.url.path == "/api/documents":
                rate_limiter.check(f"upload:{client_key(request)}", configured_limit("UPLOAD_RATE_LIMIT_PER_MINUTE", 5))
            else:
                rate_limiter.check(f"api:{client_key(request)}", configured_limit("API_RATE_LIMIT_PER_MINUTE", 120))
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/tasks", response_model=list[TaskInfo])
def list_tasks():
    return tasks()


@app.post("/api/tasks", response_model=TaskInfo, status_code=201)
def add_task(payload: TaskCreate):
    try:
        return create_task(payload.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/api/tasks/{task_id}", status_code=204)
def remove_task(task_id: str):
    try:
        delete_task(task_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")


@app.post("/api/documents", response_model=DocumentInfo)
async def upload_document(background_tasks: BackgroundTasks, task_id: str = Form(...), file: UploadFile = File(...)):
    allowed_suffixes = {".txt", ".md", ".pdf", ".docx", ".html", ".htm", ".csv", ".pptx"}
    filename = Path(file.filename or "").name
    if not filename or len(filename) > 240 or any(ord(char) < 32 for char in filename):
        raise HTTPException(400, "文件名无效")
    if Path(filename).suffix.lower() not in allowed_suffixes:
        raise HTTPException(400, "仅支持 TXT、Markdown 和 PDF 文件")
    max_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    raw = await file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(413, f"文件不能超过 {max_bytes // 1024 // 1024} MB")
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf" and not raw.startswith(b"%PDF-"):
        raise HTTPException(400, "文件扩展名为 PDF，但文件内容不是有效 PDF")
    if suffix in {".txt", ".md"}:
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "TXT 和 Markdown 文件必须使用 UTF-8 编码")
    if suffix in {".docx", ".pptx"}:
        if not raw.startswith(b"PK\x03\x04"):
            raise HTTPException(400, "Office 文件格式无效")
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if len(members) > 10_000 or sum(member.file_size for member in members) > 100 * 1024 * 1024:
                    raise HTTPException(413, "Office 文件解压后过大或包含过多文件")
        except zipfile.BadZipFile:
            raise HTTPException(400, "Office 文件格式无效")
    try:
        document = queue_ingest(task_id, filename, raw)
        background_tasks.add_task(process_ingest, task_id, document["id"])
        return document
    except KeyError:
        raise HTTPException(404, "任务不存在")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/tasks/{task_id}/documents", response_model=list[DocumentInfo])
def list_documents(task_id: str):
    try:
        return documents(task_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")


@app.delete("/api/tasks/{task_id}/documents/{document_id}", status_code=204)
def remove_document(task_id: str, document_id: str):
    try:
        delete_document(task_id, document_id)
    except KeyError:
        raise HTTPException(404, "任务或资料不存在")


@app.post("/api/tasks/{task_id}/documents/{document_id}/reindex", response_model=DocumentInfo)
def reindex(task_id: str, document_id: str, background_tasks: BackgroundTasks):
    try:
        document = reindex_document(task_id, document_id)
        background_tasks.add_task(process_ingest, task_id, document_id)
        return document
    except KeyError:
        raise HTTPException(404, "任务或资料不存在")


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        return answer(payload.message, payload.task_id, payload.session_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")


@app.post("/api/chat/stream")
def stream_chat(payload: ChatRequest):
    if not any(task["id"] == payload.task_id for task in tasks()):
        raise HTTPException(404, "任务不存在")

    def events():
        for item in stream_answer(payload.message, payload.task_id, payload.session_id):
            yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/sessions/{session_id}", response_model=list[SessionMessage])
def history(session_id: str):
    return get_session(session_id)


@app.get("/api/tasks/{task_id}/sessions/{session_id}", response_model=list[SessionMessage])
def task_history(task_id: str, session_id: str):
    if not any(task["id"] == task_id for task in tasks()):
        raise HTTPException(404, "任务不存在")
    return get_session(session_id, task_id)
