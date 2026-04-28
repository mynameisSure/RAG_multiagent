# RAG Multi-Agent Research Assistant

基于 LangChain 的多智能体 RAG 研究报告生成系统。项目面向复杂主题研究和企业私有知识库场景，支持本地文档入库、向量检索、网页检索、证据筛选、报告生成和质量评审。

> 当前项目更接近一个可演示、可扩展的研究型 RAG 原型，而不是简单的 RetrievalQA 问答链。

## 核心能力

- 多智能体编排：Supervisor 动态调度 Planner、Retriever、Evidence Curator、Writer、Critic。
- 本地知识库 RAG：支持 TXT、Markdown、PDF、DOCX、HTML 文档入库和 Chroma 向量检索。
- 网页资料补充：通过 Tavily 搜索公开资料，和本地证据统一建模。
- 证据治理：对 Evidence 进行去重、相关性、可信度和时效性评分。
- 证据驱动报告：Writer 基于筛选后的证据生成 Markdown 研究报告。
- 质量评审：Critic 检查报告是否覆盖研究问题、是否存在无证据结论。
- 多入口使用：提供 FastAPI、Typer CLI 和 Streamlit UI。
- 可观测性：记录 Agent Trace，便于复盘每一步决策。

## 系统架构

```text
用户请求
  -> FastAPI / CLI / Streamlit
  -> ResearchRequest
  -> ResearchService
  -> AgenticResearchOrchestrator
  -> Supervisor 决策下一步
  -> Planner / Local Retriever / Web Retriever / Evidence Curator / Writer / Critic
  -> ResearchReport
```

核心流程：

```text
文档入库 -> 文本切分 -> 向量化 -> Chroma
用户主题 -> 研究规划 -> 本地检索 / 网页检索
检索结果 -> Evidence -> 去重评分筛选
Evidence + QueryPlan -> Writer -> Markdown Report
Report + Evidence IDs -> Critic -> Critique
Critique -> Supervisor 决定结束或继续补充证据
```

## 目录结构

```text
RAG_multiagent/
├── agents/
│   ├── prompts.py              # Planner / Writer / Critic 系统提示词
│   └── supervisor.py           # 多智能体调度核心
├── api/
│   └── main.py                 # FastAPI 接口
├── ingestion/
│   ├── loaders.py              # 多格式文档加载
│   ├── chunker.py              # 文本切分
│   └── pipeline.py             # 入库流水线
├── retrieval/
│   ├── vector_store.py         # Chroma 向量库封装
│   ├── local_retriever.py      # 本地检索
│   ├── web_search.py           # Tavily 网页检索
│   └── evidence.py             # Evidence 去重与评分
├── services/
│   └── research_service.py     # 研究任务服务层
├── storage/
│   └── manifest.py             # 文件哈希增量入库
├── ui/
│   └── streamlit_app.py        # Streamlit 可视化界面
├── utils/
│   ├── json.py                 # LLM JSON 输出抽取
│   └── logging.py              # 结构化日志
├── models.py                   # Pydantic 数据模型
├── config.py                   # 配置管理
├── llm.py                      # LLM / Embedding 工厂
└── __main__.py                 # Typer CLI
```

## 技术栈

- Python
- LangChain 
- Chroma
- Tavily Search
- FastAPI
- Streamlit
- Typer
- Pydantic / Pydantic Settings
- BeautifulSoup
- Structlog
- Rich

## 安装

建议使用 Python 3.11+。

```powershell
git clone <your-repo-url>
cd RAG_multiagent
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
```

当前源码使用 `RAG_multiagent.*` 绝对导入。如果你在仓库根目录运行命令，请先把父目录加入 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = (Get-Item ..).FullName
```

当前仓库没有固定的 `requirements.txt` 时，可以按需安装：

```powershell
pip install langchain langchain-core langchain-community langchain-openai langchain-chroma langchain-text-splitters langchain-tavily
pip install fastapi uvicorn streamlit typer rich pydantic pydantic-settings python-dotenv beautifulsoup4 structlog pytest
pip install pypdf docx2txt
```

## 环境变量

在项目根目录创建 `.env`。

示例：

```env
RA_LLM_PROVIDER=dashscope
RA_CHAT_MODEL=qwen-plus
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

RA_EMBEDDING_PROVIDER=dashscope
RA_EMBEDDING_MODEL=text-embedding-v4

RA_VECTOR_STORE_PATH=data/vectorstore
RA_COLLECTION_NAME=research_assistant
RA_RETRIEVAL_K=8

CHUNK_SIZE=900
CHUNK_OVERLAP=150

TAVILY_API_KEY=your_tavily_api_key
```

## 使用方式

### 1. 文档入库

把本地文档放入一个本地目录，例如：

```text
data/raw/
```

然后执行：

```powershell
python -m RAG_multiagent ingest data/raw
```

支持格式：

- `.txt`
- `.md`
- `.markdown`
- `.pdf`
- `.docx`
- `.html`
- `.htm`

入库过程会进行：

```text
文件遍历 -> 哈希增量判断 -> 文档加载 -> 文本切分 -> 写入 Chroma -> 更新 Manifest
```

### 2. 命令行运行研究任务

```powershell
python -m RAG_multiagent ask "多智能体 RAG 在企业知识管理中的应用" --depth detailed --format markdown
```

禁用网页检索：

```powershell
python -m RAG_multiagent ask "企业知识库 RAG 架构" --no-web
```

保存报告：

```powershell
python -m RAG_multiagent ask "企业知识库 RAG 架构" --out data/outputs/report.md
```

### 3. 启动 FastAPI

```powershell
uvicorn RAG_multiagent.api.main:app --reload
```

健康检查：

```http
GET /health
```

研究任务：

```http
POST /research
Content-Type: application/json
```

```json
{
  "topic": "多智能体 RAG 在企业知识管理中的应用",
  "depth": "detailed",
  "report_format": "markdown",
  "language": "zh",
  "require_web": true
}
```

文档入库：

```http
POST /ingest?path=data/raw&force=false
```

### 4. 启动 Streamlit

```powershell
streamlit run ui/streamlit_app.py
```

界面支持：

- 文档入库
- 研究主题输入
- 报告展示
- Evidence 展示
- Critique 展示
- Agent Trace 展示

## 核心数据结构

### ResearchRequest

用户请求：

```json
{
  "topic": "研究主题",
  "depth": "detailed",
  "report_format": "markdown",
  "language": "zh",
  "require_web": true
}
```

### Evidence

统一表示本地检索和网页检索结果：

```json
{
  "id": "L123456",
  "source_type": "local",
  "title": "example.md",
  "content": "证据正文",
  "source": "data/raw/example.md",
  "score": 0.82,
  "metadata": {
    "chunk_index": 0,
    "query": "检索词"
  }
}
```

### ResearchReport

最终报告：

```json
{
  "title": "报告标题",
  "executive_summary": "执行摘要",
  "report_markdown": "Markdown 报告正文",
  "claims": [],
  "evidence": [],
  "agent_trace": [],
  "critique": {}
}
```

## 多智能体流程

### Supervisor

负责观察当前状态并选择下一步 action：

- `plan`
- `local_research`
- `web_research`
- `grade_evidence`
- `write_report`
- `critique_report`
- `finish`

Supervisor 通过 LangChain `StructuredTool` 进行工具选择，项目再把工具调用转换为内部调度决策。

### Planner

输入用户主题、深度、报告格式和语言，输出结构化 `QueryPlan`：

- 研究标题
- 研究意图
- 研究问题
- 本地检索词
- 网页检索词
- 期望章节
- 风险提示

### Retriever

本地检索通过 Chroma `similarity_search` 返回相关文档 chunk；网页检索通过 Tavily 返回公开资料。两类结果都会被封装成 `Evidence`。

### Evidence Curator

对 Evidence 执行：

- 去重
- 相关性评分
- 来源可信度评分
- 时效性评分
- 按最终分数排序和截断

最终分数：

```text
final_score = relevance * 0.5 + credibility * 0.3 + freshness * 0.2
```

### Writer

基于 `QueryPlan` 和 Evidence 生成 Markdown 报告。提示词要求关键事实、数字、趋势、因果判断和建议都要绑定证据。

### Critic

检查报告：

- 是否回答研究问题
- 是否存在无证据结论
- 是否引用不存在的证据 ID
- 是否需要继续检索或重写


## 项目亮点

- 不是单轮 RAG 问答，而是完整研究报告生成 workflow。
- 使用 Supervisor 动态调度，而不是固定链式 pipeline。
- 本地知识库和网页资料统一为 Evidence。
- 通过证据评分和 Critic 降低无依据生成风险。
- 提供 Agent Trace，便于解释和调试多智能体决策过程。
