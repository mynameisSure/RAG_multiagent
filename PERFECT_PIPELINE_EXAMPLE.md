# RAG Multi-Agent 完美 Pipeline 示例

本文用一个“理想情况下完整跑通”的例子，演示本项目从文档入库到最终生成研究报告的全过程。

为了让流程更容易理解，本文会明确写出：

- 用户输入是什么。
- 每一步由谁执行。
- 这一步是否调用 LLM。
- 这一步哪些能力来自 LangChain。
- 哪些逻辑是项目自己写的。
- 各个智能体实际收到的信息大概长什么样。
- 每一步输出什么。
- `AgentResearchState` 如何变化。

## 1. 示例任务

用户希望研究：

```text
多智能体 RAG 在企业知识管理中的应用落地方案
```

用户希望系统输出：

```text
一份中文 Markdown 研究报告，要求详细、专业、基于证据。
```

请求参数：

```json
{
  "topic": "多智能体 RAG 在企业知识管理中的应用落地方案",
  "depth": "detailed",
  "report_format": "markdown",
  "language": "zh",
  "require_web": true,
  "allowed_domains": [],
  "blacklist": []
}
```

## 2. 示例知识库

假设本地 `data/raw` 目录下有 3 份企业内部资料。

### 2.1 `enterprise_km_rag.md`

```markdown
# 企业知识管理中的 RAG 实践

企业知识管理系统通常面临知识分散、检索效率低、知识更新滞后、回答不可追溯等问题。
RAG 可以通过检索企业内部知识库，为大模型回答提供上下文，降低模型凭空生成的风险。
在复杂任务中，可以将流程拆分为规划、检索、证据筛选、生成、质检等多个角色，以提升系统可控性。
```

### 2.2 `agent_architecture.md`

```markdown
# 多智能体架构设计

多智能体系统可以通过 Supervisor 统一调度不同专家角色。
在研究报告生成场景中，Planner 负责拆解任务，Retriever 负责检索资料，Writer 负责生成报告，Critic 负责事实核查。
这种角色拆分可以降低单个提示词承担过多目标造成的不稳定性。
```

### 2.3 `security_governance.md`

```markdown
# 企业知识库安全治理

企业内部知识库接入大模型时，需要考虑权限隔离、敏感信息脱敏、访问审计和来源追踪。
对于涉及客户数据、财务数据和研发资料的场景，应限制检索范围，并记录用户请求、检索片段和模型输出。
```

## 3. 入库阶段

入库命令：

```powershell
python -m RAG_multiagent ingest data/raw
```

这一阶段不调用 LLM。

### 3.1 遍历文件

项目自己写的逻辑：

```python
iter_supported_files(path)
```

识别支持格式：

```python
{".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm"}
```

本例识别到：

```text
data/raw/enterprise_km_rag.md
data/raw/agent_architecture.md
data/raw/security_governance.md
```

### 3.2 Manifest 增量判断

项目自己写的逻辑：

```python
DocumentManifest.is_seen(file_path)
```

它会对文件内容计算 SHA-1 fingerprint。

示例 Manifest：

```json
{
  "F:\\langchain_heima2.0\\RAG_multiagent\\data\\raw\\enterprise_km_rag.md": {
    "sha256": "e8a0f9b0c1...",
    "chunks": 2
  }
}
```

注意：字段名叫 `sha256`，但当前代码实际使用的是 SHA-1。

如果文件内容没有变化，并且没有 `force=True`，就跳过。

本例假设 3 个文件都是新文件，所以全部进入入库。

### 3.3 文件加载成 Document

LangChain 提供：

```python
Document
```

Markdown 文件由项目自己用 `path.read_text()` 读取，然后封装成 LangChain `Document`。

示例 Document：

```json
{
  "page_content": "企业知识管理系统通常面临知识分散、检索效率低、知识更新滞后、回答不可追溯等问题...",
  "metadata": {
    "source": "data/raw/enterprise_km_rag.md",
    "file_type": ".md",
    "filename": "enterprise_km_rag.md"
  }
}
```

### 3.4 文本切分

LangChain 提供：

```python
RecursiveCharacterTextSplitter
```

项目配置：

```python
chunk_size = CHUNK_SIZE
chunk_overlap = CHUNK_OVERLAP
separators = ["\n\n", "\n", "。", "；", "，", ". ", " ", ""]
```

理想切分结果：

```json
[
  {
    "page_content": "企业知识管理系统通常面临知识分散、检索效率低、知识更新滞后、回答不可追溯等问题。RAG 可以通过检索企业内部知识库，为大模型回答提供上下文，降低模型凭空生成的风险。",
    "metadata": {
      "source": "data/raw/enterprise_km_rag.md",
      "filename": "enterprise_km_rag.md",
      "chunk_index": 0
    }
  },
  {
    "page_content": "在复杂任务中，可以将流程拆分为规划、检索、证据筛选、生成、质检等多个角色，以提升系统可控性。",
    "metadata": {
      "source": "data/raw/enterprise_km_rag.md",
      "filename": "enterprise_km_rag.md",
      "chunk_index": 1
    }
  }
]
```

### 3.5 写入 Chroma 向量库

LangChain / Chroma 提供：

```python
Chroma.add_documents(documents)
```

底层发生：

```text
chunk 文本 -> embedding 模型 -> 向量 -> Chroma 持久化
```

本项目自己控制：

```text
哪些文件入库、如何切分、metadata 怎么写、Manifest 怎么记录。
```

理想入库结果：

```json
{
  "files_seen": 3,
  "files_indexed": 3,
  "chunks_added": 6,
  "skipped": [],
  "errors": []
}
```

## 4. 研究任务开始

入口可以是 FastAPI、CLI 或 Streamlit。

例如 FastAPI 请求：

```http
POST /research
Content-Type: application/json
```

请求体：

```json
{
  "topic": "多智能体 RAG 在企业知识管理中的应用落地方案",
  "depth": "detailed",
  "report_format": "markdown",
  "language": "zh",
  "require_web": true
}
```

FastAPI 用 Pydantic 把请求转成：

```python
ResearchRequest(...)
```

然后执行：

```python
ResearchService().run(request)
```

初始 `AgentResearchState`：

```json
{
  "request": {
    "topic": "多智能体 RAG 在企业知识管理中的应用落地方案",
    "depth": "detailed",
    "report_format": "markdown",
    "language": "zh",
    "require_web": true,
    "allowed_domains": [],
    "blacklist": []
  },
  "plan": null,
  "evidence": [],
  "graded_evidence": [],
  "report": null,
  "critique": null,
  "errors": [],
  "trace": []
}
```

## 5. 第 1 轮：Supervisor 决定先规划

这一轮调用 LLM。

### 5.1 Supervisor 收到的 SystemMessage

```text
你是一个研究型多智能体系统的主管智能体。
你必须通过工具调用来调度专家能力；每一轮只调用一个最合适的工具。

可用工具：
- plan_research：让规划专家生成或重做研究计划。
- search_local_knowledge：让本地 RAG 专家检索私有知识库。
- search_web：让网页研究专家检索公开互联网资料；只有用户允许网页搜索时使用。
- curate_evidence：让证据策展专家去重、评分、筛选证据。
- write_or_revise_report：让写作专家基于证据撰写或重写报告。
- critique_report：让质检专家检查报告证据覆盖、缺口和 unsupported claims。
- finish_research：当报告已经足够好时结束。

调度原则：
1. 不要机械执行固定流水线；你要根据当前状态选择下一步工具。
2. 没有计划时通常调用 plan_research。
3. 证据不足时调用 search_local_knowledge；如果用户允许且主题需要时效性，再调用 search_web。
4. 有新证据后通常调用 curate_evidence，再考虑写作。
5. 有报告后应调用 critique_report；质检发现缺口时继续检索或重写。
6. 只有当报告存在、证据充分、质检通过或缺口可接受时才调用 finish_research。
7. 工具参数要具体，特别是检索 queries 和重写 revision_instructions。
```

### 5.2 Supervisor 收到的 HumanMessage

```text
请观察当前状态，并调用一个最合适的工具推进研究任务。
用户请求：
{
  "topic": "多智能体 RAG 在企业知识管理中的应用落地方案",
  "depth": "detailed",
  "report_format": "markdown",
  "language": "zh",
  "require_web": true,
  "allowed_domains": [],
  "blacklist": []
}

当前步数：1/10

状态摘要：
{
  "has_plan": false,
  "plan": null,
  "evidence_count": 0,
  "graded_evidence_count": 0,
  "evidence_preview": [],
  "report": null,
  "critique": null,
  "errors": [],
  "recent_trace": []
}

如果当前模型无法使用工具调用，请仅返回 JSON：
{
  "action": "plan | local_research | web_research | grade_evidence | write_report | critique_report | finish",
  "rationale": "为什么选择这一步",
  "queries": ["如需检索，给出具体查询词"],
  "revision_instructions": "如需重写报告，给出修改要求",
  "confidence": 0.0
}
```

### 5.3 Supervisor 输出

理想 tool call：

```json
{
  "name": "plan_research",
  "args": {
    "rationale": "当前任务还没有研究计划，需要先拆解研究问题、检索词和报告章节。",
    "confidence": 0.95
  }
}
```

项目自己把它转换成：

```json
{
  "action": "plan",
  "rationale": "当前任务还没有研究计划，需要先拆解研究问题、检索词和报告章节。",
  "queries": [],
  "revision_instructions": "",
  "confidence": 0.95,
  "source": "tool_call"
}
```

### 5.4 State 变化

`trace` 增加：

```json
{
  "step": 1,
  "agent": "supervisor",
  "action": "plan",
  "source": "tool_call",
  "rationale": "当前任务还没有研究计划，需要先拆解研究问题、检索词和报告章节。",
  "queries": [],
  "revision_instructions": "",
  "confidence": 0.95
}
```

## 6. 第 1 轮执行：Planner 生成研究计划

这一轮调用 LLM。

### 6.1 Planner 收到的 SystemMessage

```text
你是研究规划专家。你的任务是把用户主题拆成可检索、可验证、可交付的研究计划。

要求：
- 必须围绕用户主题生成具体、可执行的研究问题，不要泛泛而谈。
- research_questions 应覆盖：背景定义、关键事实、方法/案例、风险限制、趋势或建议。
- local_queries 面向本地知识库，适合检索内部文档、样例资料、历史知识。
- web_queries 面向公开互联网，适合检索最新资料、行业报告、政策、论文或新闻。
- 如果主题涉及医疗、法律、金融、安全、政策、实时事件，必须写入 risk_notes。
- 查询词要具体，避免只有一个宽泛关键词。
- 只返回 JSON，不要输出 Markdown 或解释文字。
```

### 6.2 Planner 收到的 HumanMessage

```text
请为下面研究任务生成 JSON 研究计划。
主题：多智能体 RAG 在企业知识管理中的应用落地方案
深度：detailed
报告格式：markdown
语言：zh

JSON schema:
{
  "title": "string",
  "intent": "string",
  "research_questions": ["string"],
  "local_queries": ["string"],
  "web_queries": ["string"],
  "expected_sections": ["string"],
  "risk_notes": ["string"]
}
```

### 6.3 Planner 理想输出

```json
{
  "title": "多智能体 RAG 在企业知识管理中的应用落地方案研究报告",
  "intent": "分析多智能体 RAG 如何解决企业知识管理中的检索、沉淀、问答可信度和治理问题，并给出可落地的系统架构与实施建议。",
  "research_questions": [
    "企业知识管理中常见的知识分散、检索效率低和回答不可追溯问题是什么？",
    "RAG 如何通过检索私有知识库降低大模型凭空生成的风险？",
    "为什么要把 RAG 工作流拆分为规划、检索、证据筛选、写作和质检等多个智能体角色？",
    "多智能体 RAG 在企业落地时需要怎样设计权限控制、脱敏、审计和来源追踪？",
    "一个可落地的多智能体 RAG 企业知识管理系统应包含哪些核心模块？"
  ],
  "local_queries": [
    "企业知识管理 RAG 知识分散 检索效率 回答不可追溯",
    "多智能体 RAG Supervisor Planner Retriever Writer Critic",
    "企业知识库 大模型 权限隔离 脱敏 审计 来源追踪"
  ],
  "web_queries": [
    "multi agent RAG enterprise knowledge management architecture",
    "retrieval augmented generation grounding citation enterprise",
    "RAG governance access control audit enterprise knowledge base"
  ],
  "expected_sections": [
    "执行摘要",
    "问题背景",
    "多智能体 RAG 架构",
    "企业落地流程",
    "安全与治理",
    "实施建议",
    "参考证据"
  ],
  "risk_notes": [
    "涉及企业内部知识库时，需要关注权限隔离、敏感信息脱敏、访问审计和来源追踪。",
    "缺少证据支持时，不应对效果提升做绝对化承诺。"
  ]
}
```

### 6.4 State 变化

`state.plan` 变为上面的 `QueryPlan`。

`trace` 增加：

```json
{
  "step": 1,
  "agent": "planner",
  "observation": "生成计划：多智能体 RAG 在企业知识管理中的应用落地方案研究报告"
}
```

## 7. 第 2 轮：Supervisor 决定本地检索

这一轮调用 LLM。

### 7.1 Supervisor 收到的状态摘要

```json
{
  "has_plan": true,
  "plan": {
    "title": "多智能体 RAG 在企业知识管理中的应用落地方案研究报告",
    "intent": "分析多智能体 RAG 如何解决企业知识管理中的检索、沉淀、问答可信度和治理问题，并给出可落地的系统架构与实施建议。",
    "research_questions": [
      "企业知识管理中常见的知识分散、检索效率低和回答不可追溯问题是什么？",
      "RAG 如何通过检索私有知识库降低大模型凭空生成的风险？",
      "为什么要把 RAG 工作流拆分为规划、检索、证据筛选、写作和质检等多个智能体角色？",
      "多智能体 RAG 在企业落地时需要怎样设计权限控制、脱敏、审计和来源追踪？",
      "一个可落地的多智能体 RAG 企业知识管理系统应包含哪些核心模块？"
    ],
    "local_queries": [
      "企业知识管理 RAG 知识分散 检索效率 回答不可追溯",
      "多智能体 RAG Supervisor Planner Retriever Writer Critic",
      "企业知识库 大模型 权限隔离 脱敏 审计 来源追踪"
    ],
    "web_queries": [
      "multi agent RAG enterprise knowledge management architecture",
      "retrieval augmented generation grounding citation enterprise",
      "RAG governance access control audit enterprise knowledge base"
    ],
    "expected_sections": [
      "执行摘要",
      "问题背景",
      "多智能体 RAG 架构",
      "企业落地流程",
      "安全与治理",
      "实施建议",
      "参考证据"
    ],
    "risk_notes": [
      "涉及企业内部知识库时，需要关注权限隔离、敏感信息脱敏、访问审计和来源追踪。",
      "缺少证据支持时，不应对效果提升做绝对化承诺。"
    ]
  },
  "evidence_count": 0,
  "graded_evidence_count": 0,
  "evidence_preview": [],
  "report": null,
  "critique": null,
  "errors": [],
  "recent_trace": [
    {
      "step": 1,
      "agent": "planner",
      "observation": "生成计划：多智能体 RAG 在企业知识管理中的应用落地方案研究报告"
    }
  ]
}
```

### 7.2 Supervisor 理想输出

```json
{
  "name": "search_local_knowledge",
  "args": {
    "queries": [
      "企业知识管理 RAG 知识分散 检索效率 回答不可追溯",
      "多智能体 RAG Supervisor Planner Retriever Writer Critic",
      "企业知识库 大模型 权限隔离 脱敏 审计 来源追踪"
    ],
    "rationale": "当前已有研究计划但没有证据，应先检索本地私有知识库，获取企业内部知识管理和系统架构相关资料。",
    "confidence": 0.9
  }
}
```

## 8. 第 2 轮执行：本地 RAG 检索

这一轮不调用 LLM。

LangChain / Chroma 负责：

```text
query -> embedding -> similarity_search -> 返回 Document
```

项目自己负责：

```text
把 Document 转成 Evidence，并写入 state.evidence。
```

### 8.1 Local Retriever 收到的信息

```python
queries = [
    "企业知识管理 RAG 知识分散 检索效率 回答不可追溯",
    "多智能体 RAG Supervisor Planner Retriever Writer Critic",
    "企业知识库 大模型 权限隔离 脱敏 审计 来源追踪"
]
```

### 8.2 Chroma 返回的 Document 示例

```json
[
  {
    "page_content": "企业知识管理系统通常面临知识分散、检索效率低、知识更新滞后、回答不可追溯等问题。RAG 可以通过检索企业内部知识库，为大模型回答提供上下文，降低模型凭空生成的风险。",
    "metadata": {
      "source": "data/raw/enterprise_km_rag.md",
      "filename": "enterprise_km_rag.md",
      "chunk_index": 0
    }
  },
  {
    "page_content": "多智能体系统可以通过 Supervisor 统一调度不同专家角色。在研究报告生成场景中，Planner 负责拆解任务，Retriever 负责检索资料，Writer 负责生成报告，Critic 负责事实核查。",
    "metadata": {
      "source": "data/raw/agent_architecture.md",
      "filename": "agent_architecture.md",
      "chunk_index": 0
    }
  },
  {
    "page_content": "企业内部知识库接入大模型时，需要考虑权限隔离、敏感信息脱敏、访问审计和来源追踪。",
    "metadata": {
      "source": "data/raw/security_governance.md",
      "filename": "security_governance.md",
      "chunk_index": 0
    }
  }
]
```

### 8.3 转换后的 Evidence

```json
[
  {
    "id": "L100001",
    "source_type": "local",
    "title": "enterprise_km_rag.md",
    "content": "企业知识管理系统通常面临知识分散、检索效率低、知识更新滞后、回答不可追溯等问题。RAG 可以通过检索企业内部知识库，为大模型回答提供上下文，降低模型凭空生成的风险。",
    "source": "data/raw/enterprise_km_rag.md",
    "score": 1.0,
    "metadata": {
      "filename": "enterprise_km_rag.md",
      "chunk_index": 0,
      "query": "企业知识管理 RAG 知识分散 检索效率 回答不可追溯"
    }
  },
  {
    "id": "L100002",
    "source_type": "local",
    "title": "agent_architecture.md",
    "content": "多智能体系统可以通过 Supervisor 统一调度不同专家角色。在研究报告生成场景中，Planner 负责拆解任务，Retriever 负责检索资料，Writer 负责生成报告，Critic 负责事实核查。",
    "source": "data/raw/agent_architecture.md",
    "score": 1.0,
    "metadata": {
      "filename": "agent_architecture.md",
      "chunk_index": 0,
      "query": "多智能体 RAG Supervisor Planner Retriever Writer Critic"
    }
  },
  {
    "id": "L100003",
    "source_type": "local",
    "title": "security_governance.md",
    "content": "企业内部知识库接入大模型时，需要考虑权限隔离、敏感信息脱敏、访问审计和来源追踪。",
    "source": "data/raw/security_governance.md",
    "score": 1.0,
    "metadata": {
      "filename": "security_governance.md",
      "chunk_index": 0,
      "query": "企业知识库 大模型 权限隔离 脱敏 审计 来源追踪"
    }
  }
]
```

### 8.4 State 变化

```json
{
  "evidence_count": 3,
  "graded_evidence_count": 0
}
```

`trace` 增加：

```json
{
  "step": 2,
  "agent": "local_researcher",
  "observation": "本地检索新增 3 条证据。"
}
```

## 9. 第 3 轮：Supervisor 决定网页检索

这一轮调用 LLM。

因为 `require_web=true`，且主题有行业落地和治理时效性，Supervisor 可以选择网页检索。

### 9.1 Supervisor 看到的 Evidence 预览

注意：Supervisor 只看到预览，不看到完整 content。

```json
{
  "evidence_count": 3,
  "evidence_preview": [
    {
      "id": "L100001",
      "type": "local",
      "title": "enterprise_km_rag.md",
      "score": 1.0
    },
    {
      "id": "L100002",
      "type": "local",
      "title": "agent_architecture.md",
      "score": 1.0
    },
    {
      "id": "L100003",
      "type": "local",
      "title": "security_governance.md",
      "score": 1.0
    }
  ],
  "report": null
}
```

### 9.2 Supervisor 理想输出

```json
{
  "name": "search_web",
  "args": {
    "queries": [
      "multi agent RAG enterprise knowledge management architecture",
      "retrieval augmented generation grounding citation enterprise",
      "RAG governance access control audit enterprise knowledge base"
    ],
    "rationale": "本地证据覆盖了基础概念和内部治理要求，但还需要公开资料补充企业级 RAG 架构、grounding 和治理实践。",
    "confidence": 0.82
  }
}
```

## 10. 第 3 轮执行：网页检索

这一轮不调用通用 LLM。

LangChain 提供：

```python
TavilySearch
```

项目调用：

```python
tool.invoke({"query": query})
```

### 10.1 Web Retriever 收到的信息

```python
queries = [
    "multi agent RAG enterprise knowledge management architecture",
    "retrieval augmented generation grounding citation enterprise",
    "RAG governance access control audit enterprise knowledge base"
]
```

### 10.2 Tavily 返回结果示例

```json
[
  {
    "title": "Enterprise RAG Systems and Grounded Generation",
    "url": "https://example.org/enterprise-rag-grounding",
    "content": "Enterprise RAG systems combine retrieval over internal knowledge sources with language model generation to ground answers in approved documents.",
    "score": 0.78
  },
  {
    "title": "Governance Patterns for AI Knowledge Bases",
    "url": "https://example.org/ai-knowledge-governance",
    "content": "Enterprise AI knowledge systems often require access control, audit trails, source attribution, and data minimization.",
    "score": 0.74
  }
]
```

### 10.3 转成 Web Evidence

```json
[
  {
    "id": "W200001",
    "source_type": "web",
    "title": "Enterprise RAG Systems and Grounded Generation",
    "content": "Enterprise RAG systems combine retrieval over internal knowledge sources with language model generation to ground answers in approved documents.",
    "url": "https://example.org/enterprise-rag-grounding",
    "source": null,
    "score": 0.78,
    "metadata": {
      "query": "retrieval augmented generation grounding citation enterprise"
    }
  },
  {
    "id": "W200002",
    "source_type": "web",
    "title": "Governance Patterns for AI Knowledge Bases",
    "content": "Enterprise AI knowledge systems often require access control, audit trails, source attribution, and data minimization.",
    "url": "https://example.org/ai-knowledge-governance",
    "source": null,
    "score": 0.74,
    "metadata": {
      "query": "RAG governance access control audit enterprise knowledge base"
    }
  }
]
```

### 10.4 State 变化

```json
{
  "evidence_count": 5,
  "graded_evidence_count": 0
}
```

## 11. 第 4 轮：Supervisor 决定证据筛选

这一轮调用 LLM 做调度，但证据筛选本身不是 LLM。

### 11.1 Supervisor 理想输出

```json
{
  "name": "curate_evidence",
  "args": {
    "rationale": "当前已有本地和网页证据，需要去重、评分和筛选，再交给写作专家。",
    "confidence": 0.9
  }
}
```

## 12. 第 4 轮执行：Evidence Curator

这一轮不调用 LLM。

项目自己写的逻辑：

```python
grade_and_filter(items, min_score, limit)
```

### 12.1 输入 Evidence

```text
L100001 企业知识管理痛点和 RAG grounding
L100002 多智能体角色拆分
L100003 权限隔离、脱敏、审计、来源追踪
W200001 Enterprise RAG grounding
W200002 AI knowledge governance
```

### 12.2 打分过程示例

以 `L100001` 为例：

```json
{
  "relevance": 1.0,
  "credibility": 0.82,
  "freshness": 0.55
}
```

最终分：

```text
1.0 * 0.5 + 0.82 * 0.3 + 0.55 * 0.2
= 0.5 + 0.246 + 0.11
= 0.856
```

以 `W200001` 为例：

```json
{
  "relevance": 0.78,
  "credibility": 0.62,
  "freshness": 0.55
}
```

最终分：

```text
0.78 * 0.5 + 0.62 * 0.3 + 0.55 * 0.2
= 0.39 + 0.186 + 0.11
= 0.686
```

### 12.3 筛选后 Evidence

```json
[
  {
    "id": "L100001",
    "score": 0.856,
    "title": "enterprise_km_rag.md"
  },
  {
    "id": "L100002",
    "score": 0.856,
    "title": "agent_architecture.md"
  },
  {
    "id": "L100003",
    "score": 0.856,
    "title": "security_governance.md"
  },
  {
    "id": "W200001",
    "score": 0.686,
    "title": "Enterprise RAG Systems and Grounded Generation"
  },
  {
    "id": "W200002",
    "score": 0.666,
    "title": "Governance Patterns for AI Knowledge Bases"
  }
]
```

### 12.4 State 变化

```json
{
  "evidence_count": 5,
  "graded_evidence_count": 5
}
```

## 13. 第 5 轮：Supervisor 决定写报告

这一轮调用 LLM。

### 13.1 Supervisor 理想输出

```json
{
  "name": "write_or_revise_report",
  "args": {
    "revision_instructions": "首次成稿。请围绕研究问题组织章节，所有实质性结论必须绑定证据，不要做无证据的效果承诺。",
    "rationale": "当前已有经过筛选的证据，可以生成初版研究报告。",
    "confidence": 0.88
  }
}
```

## 14. 第 5 轮执行：Writer 生成报告

这一轮调用 LLM。

### 14.1 Writer 收到的 SystemMessage

```text
你是严谨的证据驱动型研究报告作者。你只能使用给定 evidence block 写作。

硬性规则：
- 每个实质性事实、数字、趋势、对比、因果判断、建议都必须带证据编号，如 [L123] 或 [W456]。
- 只能引用 evidence block 中真实存在的证据 ID，禁止编造引用 ID。
- 没有证据支持的内容必须明确写成“证据不足，无法判断”。
- 如果证据之间存在冲突，必须说明冲突，并标注对应证据 ID。
- 不要捏造论文、作者、机构、URL、日期、统计数据。
- 建议部分必须能追溯到前文证据。
- 输出 Markdown。
- 报告应包含：执行摘要、关键发现、证据分析、风险与限制、建议、参考证据清单。
```

### 14.2 Writer 收到的 HumanMessage

下面是理想情况下 Writer 应该收到的信息。

注意：当前代码的 `format_evidence_for_prompt()` 还没有把 `id=` 写进 evidence block。为了构造一个“完美 pipeline 示例”，这里展示的是建议修复后的理想 evidence block，也就是每条证据明确带 `id`。

```text
研究任务：多智能体 RAG 在企业知识管理中的应用落地方案
研究意图：分析多智能体 RAG 如何解决企业知识管理中的检索、沉淀、问答可信度和治理问题，并给出可落地的系统架构与实施建议。
研究问题：
- 企业知识管理中常见的知识分散、检索效率低和回答不可追溯问题是什么？
- RAG 如何通过检索私有知识库降低大模型凭空生成的风险？
- 为什么要把 RAG 工作流拆分为规划、检索、证据筛选、写作和质检等多个智能体角色？
- 多智能体 RAG 在企业落地时需要怎样设计权限控制、脱敏、审计和来源追踪？
- 一个可落地的多智能体 RAG 企业知识管理系统应包含哪些核心模块？

期望章节：执行摘要, 问题背景, 多智能体 RAG 架构, 企业落地流程, 安全与治理, 实施建议, 参考证据
报告格式：markdown
语言：zh

可用证据：
id=L100001; source_type=local; source=data/raw/enterprise_km_rag.md; score=0.86
excerpt: 企业知识管理系统通常面临知识分散、检索效率低、知识更新滞后、回答不可追溯等问题。RAG 可以通过检索企业内部知识库，为大模型回答提供上下文，降低模型凭空生成的风险。

id=L100002; source_type=local; source=data/raw/agent_architecture.md; score=0.86
excerpt: 多智能体系统可以通过 Supervisor 统一调度不同专家角色。在研究报告生成场景中，Planner 负责拆解任务，Retriever 负责检索资料，Writer 负责生成报告，Critic 负责事实核查。

id=L100003; source_type=local; source=data/raw/security_governance.md; score=0.86
excerpt: 企业内部知识库接入大模型时，需要考虑权限隔离、敏感信息脱敏、访问审计和来源追踪。

id=W200001; source_type=web; source=https://example.org/enterprise-rag-grounding; score=0.69
excerpt: Enterprise RAG systems combine retrieval over internal knowledge sources with language model generation to ground answers in approved documents.

id=W200002; source_type=web; source=https://example.org/ai-knowledge-governance; score=0.67
excerpt: Enterprise AI knowledge systems often require access control, audit trails, source attribution, and data minimization.

上一轮质检反馈：
暂无质检反馈

主管给出的写作或修改要求：
首次成稿。请围绕研究问题组织章节，所有实质性结论必须绑定证据，不要做无证据的效果承诺。

请写出专业、可执行、证据驱动的 Markdown 报告。每个实质性事实、数字、趋势、因果判断都必须带证据编号。
```

### 14.3 Writer 理想输出

```markdown
# 多智能体 RAG 在企业知识管理中的应用落地方案研究报告

## 执行摘要

企业知识管理系统常见问题包括知识分散、检索效率低、知识更新滞后和回答不可追溯，这些问题会影响员工获取知识和复用组织经验的效率 [L100001]。RAG 通过检索企业内部知识库为大模型提供上下文，可以降低模型脱离资料凭空生成的风险 [L100001][W200001]。在复杂研究和报告生成场景中，将流程拆分为规划、检索、证据筛选、写作和质检等角色，有助于提升系统可控性和结果可审查性 [L100002]。

## 问题背景

企业知识管理并不只是保存文档，更重要的是让员工能够快速、准确、可追溯地找到知识。现有资料显示，企业知识管理系统通常面临知识分散、检索效率低、知识更新滞后和回答不可追溯等问题 [L100001]。如果直接让大模型回答企业内部问题，模型可能无法访问最新内部资料，也可能生成没有来源支撑的内容；RAG 的作用是在生成前先检索企业内部知识库，将相关资料作为上下文提供给模型 [L100001][W200001]。

## 多智能体 RAG 架构

一个适合企业知识管理的多智能体 RAG 系统可以拆分为以下角色：Supervisor 负责统一调度，Planner 负责拆解任务，Retriever 负责检索资料，Writer 负责生成报告，Critic 负责事实核查 [L100002]。这种角色拆分避免单个提示词同时承担规划、检索、写作和质检等过多目标，有助于降低流程不稳定性 [L100002]。

## 企业落地流程

建议的落地流程包括：第一，建设企业内部知识库并保留文档来源、文件名和 chunk 编号；第二，用户提交问题后由 Planner 拆解检索方向；第三，Retriever 从私有知识库召回相关片段；第四，Evidence Curator 对证据去重、评分和筛选；第五，Writer 基于证据生成报告；第六，Critic 检查报告是否存在无证据结论 [L100001][L100002]。

## 安全与治理

企业内部知识库接入大模型时，应重点考虑权限隔离、敏感信息脱敏、访问审计和来源追踪 [L100003]。公开资料也指出，企业 AI 知识系统通常需要访问控制、审计轨迹、来源归因和数据最小化机制 [W200002]。因此，多智能体 RAG 系统不应只关注回答效果，还需要在检索、生成和日志层面保留可审计记录 [L100003][W200002]。

## 实施建议

1. 优先从高频、低风险知识场景切入，例如制度问答、内部流程查询和技术文档检索，因为这些场景更容易建立可控知识库和验证结果质量 [L100001]。
2. 在架构上保留 Supervisor、Planner、Retriever、Writer、Critic 等职责分离，避免把所有任务压到单个大模型提示词中 [L100002]。
3. 对企业内部知识库建立权限隔离、脱敏和访问审计机制，尤其是客户数据、财务数据和研发资料场景 [L100003]。
4. 对生成报告要求证据引用和来源追踪，降低无证据结论进入业务决策的风险 [L100001][W200001]。

## 参考证据

- [L100001] data/raw/enterprise_km_rag.md：企业知识管理痛点与 RAG grounding。
- [L100002] data/raw/agent_architecture.md：Supervisor、Planner、Retriever、Writer、Critic 的角色拆分。
- [L100003] data/raw/security_governance.md：权限隔离、脱敏、审计和来源追踪。
- [W200001] https://example.org/enterprise-rag-grounding：企业 RAG grounding。
- [W200002] https://example.org/ai-knowledge-governance：企业 AI 知识系统治理要求。
```

### 14.4 State 变化

系统构造 `ResearchReport`：

```json
{
  "title": "多智能体 RAG 在企业知识管理中的应用落地方案研究报告",
  "executive_summary": "企业知识管理系统常见问题包括知识分散、检索效率低、知识更新滞后和回答不可追溯...",
  "claims": [
    {
      "text": "企业知识管理系统常见问题包括知识分散、检索效率低、知识更新滞后和回答不可追溯...",
      "evidence_ids": ["L100001"],
      "confidence": 0.75
    },
    {
      "text": "RAG 通过检索企业内部知识库为大模型提供上下文，可以降低模型脱离资料凭空生成的风险...",
      "evidence_ids": ["L100001", "W200001"],
      "confidence": 0.75
    }
  ],
  "evidence_count": 5
}
```

## 15. 第 6 轮：Supervisor 决定质检

这一轮调用 LLM。

### 15.1 Supervisor 看到的状态摘要

```json
{
  "has_plan": true,
  "evidence_count": 5,
  "graded_evidence_count": 5,
  "report": {
    "title": "多智能体 RAG 在企业知识管理中的应用落地方案研究报告",
    "chars": 2200,
    "claims": 2,
    "has_critique": false
  },
  "critique": null
}
```

### 15.2 Supervisor 理想输出

```json
{
  "name": "critique_report",
  "args": {
    "rationale": "报告已经生成，需要检查关键结论是否都有证据支撑，是否遗漏研究问题。",
    "confidence": 0.9
  }
}
```

## 16. 第 6 轮执行：Critic 质检

这一轮调用 LLM。

### 16.1 Critic 收到的 SystemMessage

```text
你是事实核查与质量评审专家。你的任务是检查报告是否真正被证据支撑。

检查重点：
- 是否回答了研究计划中的关键问题。
- 关键事实、数字、趋势、因果判断是否都有证据编号。
- 是否引用了不存在的证据 ID。
- 是否存在没有证据支撑的结论。
- 是否遗漏重要研究问题。
- 是否需要继续本地检索、网页检索或重写报告。

只返回 JSON。
```

### 16.2 Critic 收到的 HumanMessage

```text
请检查报告质量，并返回 JSON。
研究问题：[
  "企业知识管理中常见的知识分散、检索效率低和回答不可追溯问题是什么？",
  "RAG 如何通过检索私有知识库降低大模型凭空生成的风险？",
  "为什么要把 RAG 工作流拆分为规划、检索、证据筛选、写作和质检等多个智能体角色？",
  "多智能体 RAG 在企业落地时需要怎样设计权限控制、脱敏、审计和来源追踪？",
  "一个可落地的多智能体 RAG 企业知识管理系统应包含哪些核心模块？"
]
证据 ID：["L100001", "L100002", "L100003", "W200001", "W200002"]
已识别引用 ID：["L100001", "L100002", "L100003", "W200001", "W200002"]
启发式参考分：0.8

报告：
# 多智能体 RAG 在企业知识管理中的应用落地方案研究报告

## 执行摘要
企业知识管理系统常见问题包括知识分散、检索效率低、知识更新滞后和回答不可追溯...

返回 JSON：
{
  "passed": true,
  "quality_score": 0.0,
  "missing_evidence": ["还缺什么证据"],
  "unsupported_claims": ["哪些结论证据不足"],
  "next_queries": ["如果需要继续检索，给出查询词"],
  "notes": "简要质检说明"
}
```

### 16.3 Critic 理想输出

```json
{
  "passed": true,
  "quality_score": 0.86,
  "missing_evidence": [],
  "unsupported_claims": [],
  "next_queries": [],
  "notes": "报告覆盖了研究计划中的主要问题，关键结论均包含可识别证据 ID，未发现明显无证据结论。"
}
```

### 16.4 State 变化

```json
{
  "critique": {
    "passed": true,
    "quality_score": 0.86
  },
  "report": {
    "has_critique": true
  }
}
```

## 17. 第 7 轮：Supervisor 决定结束

这一轮调用 LLM。

### 17.1 Supervisor 看到的状态摘要

```json
{
  "has_plan": true,
  "evidence_count": 5,
  "graded_evidence_count": 5,
  "report": {
    "title": "多智能体 RAG 在企业知识管理中的应用落地方案研究报告",
    "chars": 2200,
    "claims": 2,
    "has_critique": true
  },
  "critique": {
    "passed": true,
    "quality_score": 0.86,
    "missing_evidence": [],
    "unsupported_claims": [],
    "next_queries": [],
    "notes": "报告覆盖了研究计划中的主要问题，关键结论均包含可识别证据 ID，未发现明显无证据结论。"
  }
}
```

### 17.2 Supervisor 理想输出

```json
{
  "name": "finish_research",
  "args": {
    "rationale": "报告已经生成，证据经过筛选，质检通过且没有需要继续检索的问题，可以结束研究任务。",
    "confidence": 0.93
  }
}
```

## 18. 最终输出 ResearchReport

最终返回：

```json
{
  "title": "多智能体 RAG 在企业知识管理中的应用落地方案研究报告",
  "executive_summary": "企业知识管理系统常见问题包括知识分散、检索效率低、知识更新滞后和回答不可追溯...",
  "report_markdown": "# 多智能体 RAG 在企业知识管理中的应用落地方案研究报告\n\n## 执行摘要\n...",
  "claims": [
    {
      "text": "企业知识管理系统常见问题包括知识分散、检索效率低、知识更新滞后和回答不可追溯...",
      "evidence_ids": ["L100001"],
      "confidence": 0.75
    }
  ],
  "evidence": [
    {
      "id": "L100001",
      "source_type": "local",
      "title": "enterprise_km_rag.md",
      "source": "data/raw/enterprise_km_rag.md",
      "score": 0.856
    },
    {
      "id": "L100002",
      "source_type": "local",
      "title": "agent_architecture.md",
      "source": "data/raw/agent_architecture.md",
      "score": 0.856
    }
  ],
  "agent_trace": [
    {
      "step": 1,
      "agent": "supervisor",
      "action": "plan",
      "source": "tool_call"
    },
    {
      "step": 1,
      "agent": "planner",
      "observation": "生成计划：多智能体 RAG 在企业知识管理中的应用落地方案研究报告"
    },
    {
      "step": 2,
      "agent": "supervisor",
      "action": "local_research",
      "source": "tool_call"
    },
    {
      "step": 2,
      "agent": "local_researcher",
      "observation": "本地检索新增 3 条证据。"
    },
    {
      "step": 3,
      "agent": "web_researcher",
      "observation": "网页检索新增 2 条证据。"
    },
    {
      "step": 4,
      "agent": "evidence_curator",
      "observation": "筛选后保留 5 条证据。"
    },
    {
      "step": 5,
      "agent": "writer",
      "observation": "生成报告，长度 2200 字符。"
    },
    {
      "step": 6,
      "agent": "critic",
      "observation": "质检分 0.86，passed=True。"
    }
  ],
  "critique": {
    "passed": true,
    "quality_score": 0.86,
    "missing_evidence": [],
    "unsupported_claims": [],
    "next_queries": [],
    "notes": "报告覆盖了研究计划中的主要问题，关键结论均包含可识别证据 ID，未发现明显无证据结论。"
  }
}
```

## 19. 这个例子里 LangChain 和项目代码的分工

| 步骤 | LangChain / 第三方封装 | 项目自己写的逻辑 |
| --- | --- | --- |
| 文档对象 | `Document` | 选择文件、补 metadata |
| PDF/DOCX 加载 | `PyPDFLoader`、`Docx2txtLoader` | 统一封装到 `load_file` |
| 文本切分 | `RecursiveCharacterTextSplitter` | 设置 chunk_size、overlap、separators |
| 向量库 | `Chroma` | 控制入库、检索、路径、collection |
| LLM 调用 | `ChatOpenAI` | 选择 DashScope compatible endpoint |
| 消息格式 | `SystemMessage`、`HumanMessage` | 自己拼每个 Agent 的 prompt |
| 工具调用 | `StructuredTool`、`bind_tools` | 自己定义工具、action 映射和 Guardrails |
| 网页搜索 | `TavilySearch` | 把搜索结果转 Evidence |
| 状态管理 | 无 | `AgentResearchState` |
| 编排循环 | 无 | `AgenticResearchOrchestrator.run()` |
| 证据治理 | 无 | 去重、可信度、时效性、最终评分 |
| 质检兜底 | 无 | 启发式分数、fallback critique |
| 可观测性 | 无 | `agent_trace` |

## 20. 完美 Pipeline 的关键特征

一个“完美运行”的 pipeline 应该满足：

1. 入库阶段没有错误，所有文档成功切分并写入 Chroma。
2. Supervisor 第一轮先规划，而不是直接写报告。
3. Planner 输出合法 JSON，并通过 `QueryPlan` 校验。
4. 本地检索能返回与主题相关的 chunk。
5. 网页检索只在用户允许时执行。
6. Evidence Curator 能去重、评分、筛掉低质量证据。
7. Writer 收到的 evidence block 明确包含证据 ID。
8. Writer 的每个关键结论都绑定证据 ID。
9. Critic 能识别引用 ID，并确认没有 unsupported claims。
10. Supervisor 只在报告存在且质检通过后 finish。

## 21. 当前代码要达到这个完美例子的一个小修复

当前 `retrieval/evidence.py` 的 `format_evidence_for_prompt()` 没有输出 `id`：

```python
row = (
    f"source_type={item.source_type.value}; source={item.source or item.url or 'unknown'}; "
    f"score={item.score:.2f}\n"
    f"excerpt: {item.content[:1400]}"
)
```

建议改成：

```python
row = (
    f"id={item.id}; source_type={item.source_type.value}; "
    f"source={item.source or item.url or 'unknown'}; score={item.score:.2f}\n"
    f"excerpt: {item.content[:1400]}"
)
```

这样 Writer 才能真正看到：

```text
id=L100001
```

并按要求生成：

```text
[L100001]
```

这会让 Evidence、Claim、Critic 三个环节真正闭环。
