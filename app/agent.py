"""OpenAI-compatible multi-step tool-calling loop for KnowledgePilot."""

import json
import os
from collections.abc import Callable

from .llm import chat_completion, stream_chat_completion


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search only the current task's enterprise knowledge base before answering questions about uploaded documents, policies, products, or FAQs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A focused retrieval query."},
                    "filters": {
                        "type": "object",
                        "description": "Optional metadata constraints. Only use a document name when the user explicitly names it.",
                        "properties": {
                            "document_ids": {"type": "array", "items": {"type": "string"}},
                            "documents": {"type": "array", "items": {"type": "string"}},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression when an exact calculation is needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression, for example '(12+8)*3'."},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """You are KnowledgePilot, an enterprise knowledge assistant.
Use search_knowledge_base before answering questions that may depend on current-task uploaded materials. Use calculate for arithmetic instead of calculating mentally. You may call tools repeatedly when needed. After tools return, give a concise Chinese final answer.

Citation rules: When a statement is grounded in search results, append the exact inline citation [document#chunk_id] supplied by the tool, for example [leave-policy.md#abc123]. Never invent or alter a citation. If search returns no internal material, still answer from general knowledge when appropriate, but begin the answer with: "提示：以下回答基于通用知识，没有内部资料参考。"."""


def _chat_completion(messages: list[dict]) -> tuple[dict | None, str | None]:
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
        "tool_choice": "auto",
        "temperature": 0.2,
    }
    return chat_completion(payload)


def _tool_output_for_sources(sources: list) -> str:
    return json.dumps(
        {
            "results": [
                {"document": source.document, "chunk_id": source.chunk_id, "content": source.content, "score": source.score}
                for source in sources
            ]
        },
        ensure_ascii=False,
    )


def run_agent(
    session_id: str,
    task_id: str,
    history: list[dict],
    search_fn: Callable[[str, str], tuple[list, list[str]]],
    calculate_fn: Callable[[str], str],
    completion_fn: Callable[[list[dict]], tuple[dict | None, str | None]] = _chat_completion,
) -> tuple[str | None, list, list[str], str, str | None]:
    """Run a bounded model → tool → model loop and return answer, sources, trace, route, error."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in history[-12:] if item["role"] in {"user", "assistant"})
    sources_by_id: dict[str, object] = {}
    trace = [f"加载会话上下文 {min(len(history), 12)} 条消息", "由 LLM 选择工具"]
    used_tools: list[str] = []
    max_steps = max(1, min(int(os.getenv("AGENT_MAX_TOOL_STEPS", "4")), 8))

    for step in range(max_steps + 1):
        assistant_message, error = completion_fn(messages)
        if error:
            return None, list(sources_by_id.values()), trace, "agent", error
        tool_calls = assistant_message.get("tool_calls") or []
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.get("content"),
                "tool_calls": tool_calls or None,
            }
        )
        if not tool_calls:
            content = (assistant_message.get("content") or "").strip()
            if not content:
                return None, list(sources_by_id.values()), trace, "agent", "模型未返回最终回答"
            route = "agent_knowledge" if sources_by_id else "agent_tool" if used_tools and "calculate" in used_tools else "general_llm"
            trace.append("Agent 生成最终回答")
            return content, list(sources_by_id.values()), trace, route, None
        if step == max_steps:
            return None, list(sources_by_id.values()), trace, "agent", f"工具调用超过最大轮数（{max_steps}）"

        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
                tool_result = {"error": "工具参数不是合法 JSON"}
            else:
                if name == "search_knowledge_base":
                    query = str(arguments.get("query", "")).strip()
                    if not query:
                        tool_result = {"error": "query 不能为空"}
                    else:
                        requested_filter = arguments.get("filters")
                        found, retrieval_trace = search_fn(query, task_id, metadata_filter=requested_filter) if requested_filter else search_fn(query, task_id)
                        for source in found:
                            sources_by_id[source.chunk_id] = source
                        trace.extend(retrieval_trace)
                        tool_result = json.loads(_tool_output_for_sources(found))
                elif name == "calculate":
                    try:
                        tool_result = {"result": calculate_fn(str(arguments["expression"]))}
                    except Exception as exc:
                        tool_result = {"error": f"计算失败: {exc}"}
                else:
                    tool_result = {"error": f"未注册的工具: {name}"}
            used_tools.append(name)
            trace.append(f"执行工具：{name}")
            messages.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(tool_result, ensure_ascii=False)}
            )
    return None, list(sources_by_id.values()), trace, "agent", "Agent 循环异常结束"


def run_agent_stream(
    session_id: str,
    task_id: str,
    history: list[dict],
    search_fn: Callable[[str, str], tuple[list, list[str]]],
    calculate_fn: Callable[[str], str],
):
    """Yield final-answer deltas while retaining the normal bounded tool loop."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in history[-12:] if item["role"] in {"user", "assistant"})
    sources_by_id: dict[str, object] = {}
    trace = [f"加载会话上下文 {min(len(history), 12)} 条消息", "LLM 选择工具"]
    used_tools: list[str] = []
    max_steps = max(1, min(int(os.getenv("AGENT_MAX_TOOL_STEPS", "4")), 8))

    for step in range(max_steps + 1):
        payload = {"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "messages": messages, "tools": TOOL_DEFINITIONS, "tool_choice": "auto", "temperature": 0.2}
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        started = False
        try:
            for delta in stream_chat_completion(payload):
                for call in delta.get("tool_calls") or []:
                    index = call.get("index", 0)
                    target = tool_calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    target["id"] += call.get("id", "")
                    function = call.get("function") or {}
                    target["function"]["name"] += function.get("name", "")
                    target["function"]["arguments"] += function.get("arguments", "")
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    if not tool_calls and not started:
                        started = True
                        yield {"type": "start", "has_sources": bool(sources_by_id)}
                        for part in content_parts:
                            yield {"type": "token", "content": part}
                        content_parts.clear()
                    elif not tool_calls:
                        for part in content_parts:
                            yield {"type": "token", "content": part}
                        content_parts.clear()
            tool_call_list = list(tool_calls.values())
        except Exception as exc:
            yield {"type": "error", "error": str(exc), "sources": list(sources_by_id.values()), "trace": trace}
            return

        if not tool_call_list:
            if content_parts:
                yield {"type": "start", "has_sources": bool(sources_by_id)}
                for part in content_parts:
                    yield {"type": "token", "content": part}
            route = "agent_knowledge" if sources_by_id else "agent_tool" if "calculate" in used_tools else "general_llm"
            trace.append("Agent 流式生成最终回答")
            yield {"type": "done", "sources": list(sources_by_id.values()), "trace": trace, "route": route}
            return
        if step == max_steps:
            yield {"type": "error", "error": f"工具调用超过最大轮数（{max_steps}）", "sources": list(sources_by_id.values()), "trace": trace}
            return

        messages.append({"role": "assistant", "content": None, "tool_calls": tool_call_list})
        for call in tool_call_list:
            function = call["function"]
            name = function["name"]
            try:
                arguments = json.loads(function["arguments"] or "{}")
            except json.JSONDecodeError:
                tool_result = {"error": "工具参数不是合法 JSON"}
            else:
                if name == "search_knowledge_base" and str(arguments.get("query", "")).strip():
                    requested_filter = arguments.get("filters")
                    found, retrieval_trace = search_fn(str(arguments["query"]), task_id, metadata_filter=requested_filter) if requested_filter else search_fn(str(arguments["query"]), task_id)
                    for source in found:
                        sources_by_id[source.chunk_id] = source
                    trace.extend(retrieval_trace)
                    tool_result = json.loads(_tool_output_for_sources(found))
                elif name == "calculate":
                    try:
                        tool_result = {"result": calculate_fn(str(arguments["expression"]))}
                    except Exception as exc:
                        tool_result = {"error": f"计算失败: {exc}"}
                else:
                    tool_result = {"error": f"未注册或无效的工具调用: {name}"}
            used_tools.append(name)
            trace.append(f"执行工具：{name}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(tool_result, ensure_ascii=False)})
