"""Optional LLM query rewriting with a deterministic fallback."""

import os
import re

from .llm import chat_completion


def rewrite_query(question: str) -> tuple[str, str | None]:
    """Return a retrieval-oriented query and an optional trace note.

    The original question is always preserved on failure, when no API key is configured,
    or when rewriting is disabled. This keeps retrieval available offline.
    """
    if os.getenv("QUERY_REWRITE_ENABLED", "1") != "1":
        return question, "Query Rewrite 已关闭，使用原问题"
    payload = {
        "model": os.getenv("QUERY_REWRITE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
        "temperature": 0,
        "max_tokens": 120,
        "messages": [
            {"role": "system", "content": "将用户问题改写为适合企业知识库检索的一句中文查询。保留产品名、制度名、时间、条件和数字；不要回答问题、不要解释、不要添加引号。"},
            {"role": "user", "content": question},
        ],
    }
    response, error = chat_completion(payload)
    if error or not response:
        return question, f"Query Rewrite 不可用，使用原问题：{error or '空响应'}"
    rewritten = re.sub(r"\s+", " ", str(response.get("content") or "")).strip().strip("\"'“”")
    if not rewritten or len(rewritten) > 200:
        return question, "Query Rewrite 返回无效内容，使用原问题"
    if rewritten == question:
        return question, "Query Rewrite 与原问题一致"
    return rewritten, f"Query Rewrite：{rewritten}"
