"""Resilient OpenAI-compatible Chat Completions client."""

import os
import time
import json
from collections.abc import Iterator

import httpx


def chat_completion(payload: dict) -> tuple[dict | None, str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "未检测到 OPENAI_API_KEY"
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    retries = max(0, min(int(os.getenv("LLM_MAX_RETRIES", "2")), 5))
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=min(timeout, 10))) as client:
        for attempt in range(retries + 1):
            try:
                response = client.post(f"{base}/chat/completions", headers=headers, json=payload)
                if response.status_code < 500 and response.status_code != 429:
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"], None
                error = f"模型服务返回 HTTP {response.status_code}: {response.text[:300]}"
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                error = f"模型服务网络或超时错误: {exc}"
            except (httpx.HTTPStatusError, KeyError, IndexError, ValueError) as exc:
                return None, f"模型响应失败: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
        return None, error


def stream_chat_completion(payload: dict) -> Iterator[dict]:
    """Yield OpenAI-compatible Chat Completions SSE deltas."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 OPENAI_API_KEY")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    stream_payload = {**payload, "stream": True}
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=min(timeout, 10))) as client:
        with client.stream("POST", f"{base}/chat/completions", headers=headers, json=stream_payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    choice = json.loads(data)["choices"][0]
                    yield choice.get("delta", {})
                except (ValueError, KeyError, IndexError):
                    continue
