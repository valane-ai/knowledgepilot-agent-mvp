# KnowledgePilot Agent

## Agent 运行方式

项目使用 OpenAI 兼容的 Chat Completions Function Calling 实现有限步 Agent 循环：模型读取最近会话历史，自主选择工具，工具结果回填给模型后再生成最终回答。当前内置 `search_knowledge_base`（仅检索当前任务资料）和 `calculate` 两个工具；后续可在同一工具注册表中接入 Web 搜索、SQL 或企业内部 API。

每轮最多调用次数由 `.env` 中的 `AGENT_MAX_TOOL_STEPS` 控制，默认值为 `4`。未配置模型或模型不支持工具调用时，系统自动降级为本地 RAG 资料回答。

## 存储与可靠性

运行时主数据使用 `data/knowledgepilot.db`（SQLite），开启 WAL、外键和事务保护；首次运行会从旧 JSON 文件迁移任务、资料、切块和会话，旧 JSON 保留为备份。文档删除在一个数据库事务中清理资料与切块，文件删除采用临时文件与补偿清理策略。上传限制默认 20 MB，并校验扩展名、PDF 文件头和文本 UTF-8 编码；上传后先返回 `processing`，再由 FastAPI 后台任务完成解析、切块与索引。

模型客户端使用 `httpx`，支持连接池、可配置超时和对网络错误、429、5xx 的有限指数退避重试。可通过 `.env` 设置 `MAX_UPLOAD_BYTES`、`LLM_TIMEOUT_SECONDS`、`LLM_MAX_RETRIES`。

## 安全与质量门禁

- 可选设置 `KNOWLEDGEPILOT_API_KEYS`（逗号分隔）启用所有 `/api/*` 的 `X-API-Key` 校验；未设置时保留本地开发模式。
- 内存限流默认：聊天 20 次/分钟、上传 5 次/分钟、其他 API 120 次/分钟。多实例部署时应替换为 Redis 限流。
- 请求模型拒绝额外字段与控制字符；Office 上传校验 ZIP 结构、条目数和解压大小。生产环境仍建议接入网关、ClamAV 与企业 SSO。
- 计算器使用递归 AST 解释器，不再调用 `eval()`，并限制表达式复杂度、指数和结果范围。
- `evals/run_retrieval_eval.py` 可输出 Recall@K 与 MRR；使用真实任务资料补充 `evals/qa_dataset.jsonl` 后运行评测。

一个可本地运行的企业知识库问答 Agent。它支持上传文本类文档、检索相关内容、调用计算器工具，并返回可追溯的引用来源。

## 功能

- 文档上传与自动切分（TXT、MD、PDF）
- 本地 BM25 风格关键词检索，无需额外数据库即可演示
- 有状态 Tool Calling Agent：模型可调用当前任务检索和计算工具，并在有限步循环后生成答案
- 答案内联引用：基于内部资料的陈述使用 `[文档名#片段ID]`，未命中内部资料时明确标记为通用大模型回答
- 引用一致性校验：检查引用是否来自本次检索结果，并提示较长的未引用陈述
- 多轮会话记录、检索轨迹和引用卡片
- 回答支持安全的 Markdown 展示；默认通过 `POST /api/chat/stream` 以 SSE 推送生成内容，保留 `POST /api/chat` 供 API 客户端兼容使用。
- 会话按知识任务保存并在切换回来时恢复，接口为 `GET /api/tasks/{task_id}/sessions/{session_id}`；来源卡片可展开查看检索片段正文。
- 可选接入兼容 OpenAI 的模型 API；未配置时使用可演示的规则化回答

## 文档处理与知识任务管理

- 支持 `TXT`、`Markdown`、`PDF`、`DOCX`、`HTML`、`CSV` 和 `PPTX`；上传后会进入后台解析，界面会显示处理进度或失败原因。
- 文本按 Markdown 标题、段落和较长段落的自然边界切分，并保留标题上下文，替代固定字数硬切块。
- 检索前可用 LLM 将口语问题改写为检索查询（`QUERY_REWRITE_ENABLED=1`）；失败或未配置模型时自动使用原问题。系统会自动识别用户点名的文件名，并以文档元数据过滤结果。
- 检索结果不再固定 top-k：根据重排分数的相邻跌落动态选择 2～8 个证据片段，可通过 `RAG_DYNAMIC_*` 环境变量调整。
- PDF 默认用 `pypdf` 提取文本；文本不足时会使用 `pdfplumber` 提取页面文本及表格。扫描件可启用 Tesseract OCR：安装 Tesseract 与 Poppler 后设置 `ENABLE_OCR=1`，可选设置 `OCR_LANG=chi_sim+eng`。
- 每份资料可在界面中点击“↻”重新解析与重建索引。删除知识任务会级联删除该任务的资料、切块、已归属的历史对话和上传文件。

## 快速启动

```powershell
conda env create -f environment.yml
conda activate knowledgepilot
uvicorn app.main:app --reload
```

浏览器访问 `http://127.0.0.1:8000`。API 文档位于 `/docs`。

如果已手动创建了环境，也可以使用：

```powershell
conda create -n knowledgepilot python=3.11 -y
conda activate knowledgepilot
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 可选模型配置

复制 `.env.example` 为 `.env` 并填写模型服务配置。模型配置不是启动本项目的必要条件；未配置时会采用本地降级回答。

```powershell
Copy-Item .env.example .env
```

`.env` 使用 OpenAI 兼容的 Chat Completions 协议，支持以下常见服务：

| 服务 | `OPENAI_BASE_URL` 示例 | `OPENAI_MODEL` 示例 |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4.1-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 阿里云百炼（通义） | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |

修改 `.env` 后请重启 `uvicorn`。模型回答仍只使用本次检索到的知识库片段，前端会在调用轨迹中标记“调用 LLM 生成答案”。

## 项目结构

```text
app/
  main.py          # API 与静态页面
  services.py      # 文档处理、检索、Agent 编排
  schemas.py       # 请求与响应模型
  static/          # 前端页面
data/              # 运行时生成的文档、索引与会话数据
tests/             # 核心流程测试
```
