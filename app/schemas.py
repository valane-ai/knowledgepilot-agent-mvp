import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean_text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name}不能为空")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError(f"{field_name}包含不允许的控制字符")
    return value


class Source(BaseModel):
    document: str
    chunk_id: str
    content: str
    score: float


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    task_id: str = Field(min_length=1, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        return _clean_text(value, "消息")

    @field_validator("task_id", "session_id")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("标识符格式无效")
        return value


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]
    faithfulness: "FaithfulnessReport"
    route: str
    trace: list[str]


class FaithfulnessReport(BaseModel):
    status: str
    message: str
    citations: list[str] = []
    invalid_citations: list[str] = []
    uncited_statements: list[str] = []


ChatResponse.model_rebuild()


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunks: int
    status: str = "ready"
    progress: int = 100
    error_detail: str | None = None


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        return _clean_text(value, "任务名称")


class TaskInfo(BaseModel):
    id: str
    name: str
    document_count: int
    chunk_count: int


class SessionMessage(BaseModel):
    role: str
    content: str
